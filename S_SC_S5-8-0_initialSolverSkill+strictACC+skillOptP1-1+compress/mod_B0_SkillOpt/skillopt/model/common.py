"""Shared model utilities for ReflACT backends."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


_RESPONSES_API_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.1-codex",
    "gpt-5.2-codex",
    "gpt-5-codex",
    "codex-mini",
    "gpt-5.4-pro",
}

_BACKEND_DEFAULT_MODELS = {
    "azure_openai": "gpt-4o",
    "openai_chat": "gpt-4o",
    "codex": "gpt-4o",
    "codex_exec": "gpt-4o",
    "claude": "claude-sonnet-4-6",
    "claude_chat": "claude-sonnet-4-6",
    "claude_code_exec": "claude-sonnet-4-6",
    "qwen_chat": "Qwen/Qwen3.5-4B",
    "minimax_chat": "MiniMax-M2.7",
}

_BACKEND_ALIASES = {
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "azure-openai": "azure_openai",
    "openai_chat": "openai_chat",
    "openai": "codex",
    "codex": "codex",
    "codex_exec": "codex_exec",
    "claude": "claude_chat",
    "claude_chat": "claude_chat",
    "claude_code_exec": "claude_code_exec",
    "anthropic": "claude_chat",
    "qwen": "qwen_chat",
    "qwen_chat": "qwen_chat",
    "minimax": "minimax_chat",
    "minimax_chat": "minimax_chat",
}

_TOKEN_LENGTH_RECORD_LOCK = threading.Lock()
_TOKEN_LENGTH_EXCEEDED_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
}


def _token_length_record_path() -> str:
    return (
        os.environ.get("SPQ_TOKEN_LENGTH_RECORD_PATH")
        or os.environ.get("SKILLOPT_TOKEN_LENGTH_RECORD_PATH")
        or ""
    ).strip()


def _token_length_exceeded(finish_reason: Any) -> bool:
    reason = str(finish_reason or "").strip().lower()
    if reason in _TOKEN_LENGTH_EXCEEDED_REASONS:
        return True
    return "token" in reason and ("max" in reason or "limit" in reason)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_token_length_event(
    *,
    source: str,
    stage: str,
    model: str | None = None,
    request_type: str | None = None,
    finish_reason: Any = None,
    max_tokens_configured: Any = None,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    output_chars: int | None = None,
) -> None:
    """Append one raw token-length record without affecting model flow."""
    path = _token_length_record_path()
    if not path:
        return

    prompt_i = _optional_int(prompt_tokens)
    completion_i = _optional_int(completion_tokens)
    total_i = _optional_int(total_tokens)
    max_i = _optional_int(max_tokens_configured)
    if total_i is None and prompt_i is not None and completion_i is not None:
        total_i = prompt_i + completion_i

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "stage": stage,
        "model": model,
        "request_type": request_type,
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "token_length_exceeded": _token_length_exceeded(finish_reason),
        "max_tokens_configured": max_i,
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
        "output_chars": output_chars,
        "near_configured_limit": (
            bool(max_i and completion_i is not None and completion_i >= int(max_i * 0.95))
        ),
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _TOKEN_LENGTH_RECORD_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return


def normalize_backend_name(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    return _BACKEND_ALIASES.get(normalized, normalized or "azure_openai")


def default_model_for_backend(backend: str | None) -> str:
    return _BACKEND_DEFAULT_MODELS.get(
        normalize_backend_name(backend),
        _BACKEND_DEFAULT_MODELS["azure_openai"],
    )


def needs_responses_api(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return any(
        normalized == prefix or normalized.startswith(prefix + "-")
        for prefix in _RESPONSES_API_MODELS
    )


class TokenTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = {}

    def record(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        max_tokens_configured: Any = None,
        finish_reason: Any = None,
        output_chars: int | None = None,
        source: str = "B0",
        model: str | None = None,
        request_type: str | None = None,
    ) -> None:
        with self._lock:
            if stage not in self._data:
                self._data[stage] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            entry = self._data[stage]
            entry["calls"] += 1
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
        record_token_length_event(
            source=source,
            stage=stage,
            model=model,
            request_type=request_type,
            finish_reason=finish_reason,
            max_tokens_configured=max_tokens_configured,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_chars=output_chars,
        )

    def summary(self) -> dict[str, dict[str, int]]:
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            total_prompt = total_completion = total_calls = 0
            for stage, entry in sorted(self._data.items()):
                prompt_tokens = entry["prompt_tokens"]
                completion_tokens = entry["completion_tokens"]
                out[stage] = {
                    "calls": entry["calls"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                total_prompt += prompt_tokens
                total_completion += completion_tokens
                total_calls += entry["calls"]
            out["_total"] = {
                "calls": total_calls,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            }
            return out

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


tracker = TokenTracker()


@dataclass
class CompatToolFunction:
    name: str
    arguments: str

    def model_dump(self, mode: str = "json") -> dict[str, str]:
        del mode
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class CompatToolCall:
    id: str
    function: CompatToolFunction
    type: str = "function"

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass
class CompatAssistantMessage:
    content: str
    tool_calls: list[CompatToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        return data


def usage_from_openai_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def usage_from_responses_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def compat_message_from_chat_message(message: Any) -> CompatAssistantMessage:
    content = getattr(message, "content", "") or ""
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        tool_calls.append(
            CompatToolCall(
                id=getattr(tool_call, "id", "") or "",
                function=CompatToolFunction(
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", "") or "{}",
                ),
            )
        )
    return CompatAssistantMessage(content=content, tool_calls=tool_calls)


def compat_message_from_responses_output(output: list[Any]) -> CompatAssistantMessage:
    text_parts: list[str] = []
    tool_calls: list[CompatToolCall] = []
    for item in output:
        item_type = getattr(item, "type", "") or ""
        if item_type == "function_call":
            raw_arguments = getattr(item, "arguments", None)
            if raw_arguments is None:
                raw_arguments = json.dumps(getattr(item, "input", {}) or {})
            tool_calls.append(
                CompatToolCall(
                    id=getattr(item, "call_id", "") or getattr(item, "id", "") or "",
                    function=CompatToolFunction(
                        name=getattr(item, "name", "") or "",
                        arguments=str(raw_arguments or "{}"),
                    ),
                )
            )
            continue
        if item_type != "message":
            continue
        for part in getattr(item, "content", []) or []:
            part_type = getattr(part, "type", "") or ""
            if part_type in {"output_text", "text"}:
                text_parts.append(getattr(part, "text", "") or "")
    return CompatAssistantMessage(content="".join(text_parts), tool_calls=tool_calls)
