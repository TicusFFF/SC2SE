from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional


DEFAULT_MAX_LLM_CALLS = 3200
DEFAULT_MAX_PYTHON_TOOL_CALLS = 1600
# Keep the legacy parameter name; only confirmed GPU queue waits are deducted.
DEFAULT_MAX_WALL_SECONDS = 21600

_SAMPLE_ID: ContextVar[Optional[str]] = ContextVar(
    "spq_operation_budget_sample_id",
    default=None,
)
_SAMPLE_STARTED_AT: ContextVar[Optional[float]] = ContextVar(
    "spq_operation_budget_started_at",
    default=None,
)
_SAMPLE_GPU_QUEUE_SECONDS: ContextVar[float] = ContextVar(
    "spq_operation_budget_gpu_queue_seconds",
    default=0.0,
)
_LOCK = threading.Lock()
_COUNTS: Dict[str, Dict[str, int]] = {}


def gpu_queue_client_kwargs(base_url: str) -> Dict[str, object]:
    """Account only for waits reported by this entry's local GPU proxy.

    P1-1 pauses its clock in the calling process. S5 queues in a separate
    proxy process, so its measured wait is returned in an HTTP header.
    No configured proxy => keep the original client and timing unchanged.
    """
    proxy_url = os.environ.get("SPQ_GPU_QUEUE_PROXY_BASE_URL", "").rstrip("/")
    if not proxy_url or str(base_url).rstrip("/") != proxy_url:
        return {}

    from openai import DefaultHttpxClient

    def mark_request(request):
        request.extensions["spq_gpu_queue_clock"] = (
            time.monotonic(), _SAMPLE_STARTED_AT.get(),
        )

    def record_response(response):
        # Hooks run before status checking/stream consumption, including 4xx/5xx.
        if response.extensions.get("spq_gpu_queue_recorded"):
            return
        response.extensions["spq_gpu_queue_recorded"] = True
        if str(response.request.url).rstrip("/") not in {
            proxy_url + "/chat/completions", proxy_url + "/completions",
        }:
            return
        started, sample_started = response.request.extensions.get(
            "spq_gpu_queue_clock", (None, None),
        )
        if started is None or sample_started is None or sample_started != _SAMPLE_STARTED_AT.get():
            return
        try:
            queued = float(response.headers.get("X-SPQ-GPU-Queue-Seconds", ""))
        except (TypeError, ValueError):
            return
        elapsed = max(0.0, time.monotonic() - started)
        if not math.isfinite(queued) or queued < 0 or queued > elapsed + 0.01:
            return
        _SAMPLE_GPU_QUEUE_SECONDS.set(
            _SAMPLE_GPU_QUEUE_SECONDS.get() + min(queued, elapsed)
        )

    return {"http_client": DefaultHttpxClient(event_hooks={
        "request": [mark_request], "response": [record_response],
    })}


def _positive_limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def configured_limits() -> Dict[str, int]:
    return {
        "llm_calls": _positive_limit(
            "SPQ_SAMPLE_MAX_LLM_CALLS",
            DEFAULT_MAX_LLM_CALLS,
        ),
        "python_tool_calls": _positive_limit(
            "SPQ_SAMPLE_MAX_PYTHON_TOOL_CALLS",
            DEFAULT_MAX_PYTHON_TOOL_CALLS,
        ),
        "wall_seconds": _positive_limit(
            "SPQ_SAMPLE_MAX_WALL_SECONDS",
            DEFAULT_MAX_WALL_SECONDS,
        ),
    }


class SampleBudgetExceeded(BaseException):
    def __init__(
        self,
        *,
        sample_id: str,
        operation: str,
        limit: int,
        counts: Dict[str, int],
        elapsed_wall_seconds: float = 0.0,
        gpu_queue_seconds: float = 0.0,
    ) -> None:
        self.sample_id = str(sample_id)
        self.operation = str(operation)
        self.limit = int(limit)
        self.counts = dict(counts)
        self.elapsed_wall_seconds = float(elapsed_wall_seconds)
        self.gpu_queue_seconds = float(gpu_queue_seconds)
        self.stop_reason = {
            "llm_calls": "sample_llm_call_limit_reached",
            "python_tool_calls": "sample_python_tool_call_limit_reached",
            "wall_seconds": "sample_wall_time_limit_reached",
        }.get(self.operation, "sample_budget_exceeded")
        super().__init__(
            f"Sample {self.sample_id} reached the {self.operation} "
            f"limit ({self.limit})."
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "operation": self.operation,
            "limit": self.limit,
            "counts": dict(self.counts),
            "elapsed_wall_seconds": round(self.elapsed_wall_seconds, 3),
            "gpu_queue_seconds": round(self.gpu_queue_seconds, 3),
            "elapsed_execution_seconds": round(
                max(0.0, self.elapsed_wall_seconds - self.gpu_queue_seconds), 3,
            ),
            "stop_reason": self.stop_reason,
            "limits": configured_limits(),
        }


def _elapsed_wall_seconds() -> float:
    started_at = _SAMPLE_STARTED_AT.get()
    if started_at is None:
        return 0.0
    return max(0.0, time.monotonic() - started_at)


def check_sample_budget() -> None:
    sample_id = _SAMPLE_ID.get()
    if sample_id is None:
        return

    limit = configured_limits()["wall_seconds"]
    elapsed = _elapsed_wall_seconds()
    # Only resource-queue waits are excluded; inference/tools/retry backoff count.
    if max(0.0, elapsed - _SAMPLE_GPU_QUEUE_SECONDS.get()) < limit:
        return

    with _LOCK:
        counts = dict(
            _COUNTS.get(
                sample_id,
                {"llm_calls": 0, "python_tool_calls": 0},
            )
        )
    raise SampleBudgetExceeded(
        sample_id=sample_id,
        operation="wall_seconds",
        limit=limit,
        counts=counts,
        elapsed_wall_seconds=elapsed,
        gpu_queue_seconds=_SAMPLE_GPU_QUEUE_SECONDS.get(),
    )


@contextmanager
def sample_budget_scope(sample_id: str) -> Iterator[None]:
    sample_text = str(sample_id)
    sample_token = _SAMPLE_ID.set(sample_text)
    started_token = _SAMPLE_STARTED_AT.set(time.monotonic())
    queue_token = _SAMPLE_GPU_QUEUE_SECONDS.set(0.0)
    with _LOCK:
        _COUNTS.setdefault(
            sample_text,
            {"llm_calls": 0, "python_tool_calls": 0},
        )
    try:
        check_sample_budget()
        yield
        check_sample_budget()
    finally:
        _SAMPLE_GPU_QUEUE_SECONDS.reset(queue_token)
        _SAMPLE_STARTED_AT.reset(started_token)
        _SAMPLE_ID.reset(sample_token)


def sample_budget_snapshot(sample_id: Optional[str] = None) -> Dict[str, object]:
    sample_text = str(sample_id) if sample_id is not None else _SAMPLE_ID.get()
    if sample_text is None:
        return {
            "sample_id": None,
            "counts": {},
            "elapsed_wall_seconds": 0.0,
            "gpu_queue_seconds": 0.0,
            "elapsed_execution_seconds": 0.0,
            "limits": configured_limits(),
        }
    with _LOCK:
        counts = dict(
            _COUNTS.get(
                sample_text,
                {"llm_calls": 0, "python_tool_calls": 0},
            )
        )
    elapsed = _elapsed_wall_seconds() if sample_text == _SAMPLE_ID.get() else 0.0
    queued = _SAMPLE_GPU_QUEUE_SECONDS.get() if sample_text == _SAMPLE_ID.get() else 0.0
    return {
        "sample_id": sample_text,
        "counts": counts,
        "elapsed_wall_seconds": round(elapsed, 3),
        "gpu_queue_seconds": round(queued, 3),
        "elapsed_execution_seconds": round(max(0.0, elapsed - queued), 3),
        "limits": configured_limits(),
    }


def _consume(operation: str) -> None:
    sample_id = _SAMPLE_ID.get()
    if sample_id is None:
        return

    check_sample_budget()

    limits = configured_limits()
    limit = limits[operation]
    with _LOCK:
        counts = _COUNTS.setdefault(
            sample_id,
            {"llm_calls": 0, "python_tool_calls": 0},
        )
        if counts[operation] >= limit:
            raise SampleBudgetExceeded(
                sample_id=sample_id,
                operation=operation,
                limit=limit,
                counts=counts,
            )
        counts[operation] += 1


def consume_llm_call() -> None:
    _consume("llm_calls")


def consume_python_tool_call() -> None:
    _consume("python_tool_calls")
