from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from skillopt.model import (
    chat_optimizer,
)
from skillopt.prompts import (
    load_prompt,
)
from skillopt.utils import (
    extract_json,
)

try:
    from utils.sample_budget import (
        SampleBudgetExceeded,
        sample_budget_scope,
    )
except ImportError:
    class SampleBudgetExceeded(BaseException):
        pass

    def sample_budget_scope(sample_id):
        return nullcontext()


_ALLOWED_ROUTE_RELATION = {
    "same_route",
    "related_route",
    "different_valid_route",
    "different_invalid_route",
    "unclear",
}

_ALLOWED_OLD_ERROR_RESOLVED = {
    "yes",
    "partially",
    "no",
    "unclear",
}

_ALLOWED_ERROR_SHIFT = {
    "fixed",
    "later",
    "same",
    "earlier",
    "different_route_better",
    "different_route_worse",
    "unclear",
}

_ALLOWED_CONFIDENCE = {
    "low",
    "medium",
    "high",
}


def _clip(
    value: Any,
    max_chars: int | None = None,
) -> str:
    """Render complete evidence; max_chars is compatibility-only."""
    del max_chars
    return str(
        value
        if value is not None
        else ""
    )


def _as_bool_text(
    value: Any,
) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def _result_id(
    result: Dict[str, Any],
) -> str:
    return str(
        result.get("id")
        or ""
    )


def _index_results(
    results: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if not isinstance(result, dict):
            continue

        sample_id = _result_id(
            result
        )

        if not sample_id:
            continue

        if sample_id in indexed:
            raise ValueError(
                f"Duplicate sample id in selection results: {sample_id}"
            )

        indexed[sample_id] = result

    return indexed


def _format_forward_step(
    step: Dict[str, Any],
    *,
    index: int,
) -> str:
    parts = [
        f"[Forward Step {index}]",
        f"r_f: {_as_bool_text(step.get('r_f'))}",
    ]

    V_f = str(
        step.get("V_f")
        or ""
    ).strip()

    if V_f:
        parts.extend(
            [
                "",
                "V_f:",
                _clip(
                    V_f,
                    1200,
                ),
            ]
        )

    R_f = str(
        step.get("R_f")
        or ""
    ).strip()

    if R_f:
        parts.extend(
            [
                "",
                "R_f:",
                _clip(
                    R_f,
                    2200,
                ),
            ]
        )

    F_f = str(
        step.get("F_f")
        or ""
    ).strip()

    if F_f:
        parts.extend(
            [
                "",
                "F_f:",
                _clip(
                    F_f,
                    2200,
                ),
            ]
        )

    y_prime = str(
        step.get("y_prime")
        or ""
    ).strip()

    if y_prime:
        parts.extend(
            [
                "",
                "y_prime:",
                _clip(
                    y_prime,
                    1800,
                ),
            ]
        )

    a_prime = str(
        step.get("a_prime")
        or ""
    ).strip()

    if a_prime:
        parts.extend(
            [
                "",
                "a_prime:",
                _clip(
                    a_prime,
                    800,
                ),
            ]
        )

    return "\n".join(
        parts
    )


def _format_result_material(
    *,
    label: str,
    result: Dict[str, Any],
) -> str:
    parts = [
        f"[{label} Initial Source]",
        str(
            result.get("initial_response_source")
            or "unknown"
        ),
        "",
        f"[{label} Initial Answer]",
        _clip(
            result.get("initial_answer", ""),
            800,
        ),
        "",
        f"[{label} Initial Response]",
        _clip(
            result.get("initial_response", ""),
            5000,
        ),
        "",
        f"[{label} Forward Summary]",
        f"trajectory_type: {result.get('trajectory_type', '')}",
        f"hard: {result.get('hard', '')}",
        f"soft: {result.get('soft', '')}",
        f"forward_initial_pass: {result.get('forward_initial_pass', '')}",
        f"forward_final_pass: {result.get('forward_final_pass', '')}",
        f"forward_repair_attempted: {result.get('forward_repair_attempted', '')}",
        f"forward_repair_accepted: {result.get('forward_repair_accepted', '')}",
    ]

    fail_reason = str(
        result.get("fail_reason")
        or ""
    ).strip()

    if fail_reason:
        parts.extend(
            [
                "",
                f"[{label} Fail Reason]",
                _clip(
                    fail_reason,
                    2200,
                ),
            ]
        )

    steps = result.get("forward_steps") or []

    if isinstance(steps, list):
        for index, step in enumerate(
            steps,
            start=1,
        ):
            if not isinstance(step, dict):
                continue

            parts.extend(
                [
                    "",
                    _format_forward_step(
                        step,
                        index=index,
                    ),
                ]
            )

    final_response = str(
        result.get("final_response")
        or ""
    ).strip()

    if final_response:
        parts.extend(
            [
                "",
                f"[{label} Final Response]",
                _clip(
                    final_response,
                    3000,
                ),
            ]
        )

    final_answer = str(
        result.get("final_answer")
        or ""
    ).strip()

    if final_answer:
        parts.extend(
            [
                "",
                f"[{label} Final Answer]",
                _clip(
                    final_answer,
                    800,
                ),
            ]
        )

    return "\n".join(
        parts
    )


def build_contrastive_user_prompt(
    *,
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
) -> str:
    problem = (
        new_result.get("task_description")
        or old_result.get("task_description")
        or ""
    )

    return "\n\n".join(
        [
            "[Problem]",
            _clip(
                problem,
                5000,
            ),
            _format_result_material(
                label="Old",
                result=old_result,
            ),
            _format_result_material(
                label="New",
                result=new_result,
            ),
        ]
    )


def _as_int(
    value: Any,
    *,
    default: int,
) -> int:
    try:
        return int(
            value
        )
    except (TypeError, ValueError):
        return default


def _clamp_int(
    value: Any,
    *,
    low: int,
    high: int,
    default: int,
) -> int:
    parsed = _as_int(
        value,
        default=default,
    )

    return max(
        low,
        min(
            high,
            parsed,
        ),
    )


def _allowed_string(
    value: Any,
    *,
    allowed: set[str],
    default: str,
) -> str:
    text = str(
        value
        or ""
    ).strip()

    if text in allowed:
        return text

    return default


def _as_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    text = str(
        value
        or ""
    ).strip().lower()

    return text in {
        "1",
        "true",
        "yes",
        "y",
    }


def _as_string_list(
    value: Any,
) -> List[str]:
    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    text = str(
        value
        or ""
    ).strip()

    if not text:
        return []

    return [
        text
    ]


def _normalise_judgment(
    *,
    payload: Dict[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    delta = _clamp_int(
        payload.get("delta"),
        low=-2,
        high=2,
        default=0,
    )

    confidence = _allowed_string(
        payload.get("confidence"),
        allowed=_ALLOWED_CONFIDENCE,
        default="low",
    )

    format_only_change = _as_bool(
        payload.get("format_only_change")
    )

    # Never allow a positive score for formatting-only change.
    if format_only_change and delta > 0:
        delta = 0

    return {
        "id": sample_id,
        "delta": delta,
        "old_progress_level": _clamp_int(
            payload.get("old_progress_level"),
            low=0,
            high=5,
            default=0,
        ),
        "new_progress_level": _clamp_int(
            payload.get("new_progress_level"),
            low=0,
            high=5,
            default=0,
        ),
        "route_relation": _allowed_string(
            payload.get("route_relation"),
            allowed=_ALLOWED_ROUTE_RELATION,
            default="unclear",
        ),
        "old_error_resolved": _allowed_string(
            payload.get("old_error_resolved"),
            allowed=_ALLOWED_OLD_ERROR_RESOLVED,
            default="unclear",
        ),
        "error_shift": _allowed_string(
            payload.get("error_shift"),
            allowed=_ALLOWED_ERROR_SHIFT,
            default="unclear",
        ),
        "substantive_improvement": _as_bool(
            payload.get("substantive_improvement")
        ),
        "format_only_change": format_only_change,
        "confidence": confidence,
        "old_first_substantive_error": str(
            payload.get("old_first_substantive_error")
            or ""
        ),
        "new_first_substantive_error": str(
            payload.get("new_first_substantive_error")
            or ""
        ),
        "evidence_used": _as_string_list(
            payload.get("evidence_used")
        ),
        "justification": str(
            payload.get("justification")
            or ""
        ),
    }


def judge_contrastive_pair(
    *,
    sample_id: str,
    old_result: Dict[str, Any],
    new_result: Dict[str, Any],
    system_prompt: str,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    user_prompt = build_contrastive_user_prompt(
        old_result=old_result,
        new_result=new_result,
    )

    response, usage = chat_optimizer(
        system=system_prompt,
        user=user_prompt,
        max_completion_tokens=max_completion_tokens,
        retries=4,
        stage="contrastive_soft",
    )

    text = str(
        response
    ).strip()

    payload = extract_json(
        text
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "contrastive_soft judge returned non-JSON response: "
            f"{text[:500]!r}"
        )

    record = _normalise_judgment(
        payload=payload,
        sample_id=sample_id,
    )

    record["usage"] = usage

    return record


def _safe_error_record(
    *,
    sample_id: str,
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "delta": 0,
        "old_progress_level": 0,
        "new_progress_level": 0,
        "route_relation": "unclear",
        "old_error_resolved": "unclear",
        "error_shift": "unclear",
        "substantive_improvement": False,
        "format_only_change": False,
        "confidence": "low",
        "old_first_substantive_error": "",
        "new_first_substantive_error": "",
        "evidence_used": [],
        "justification": "",
    }


def _write_jsonl(
    *,
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            for row in rows
        ),
        encoding="utf-8",
    )


def _count_by(
    rows: List[Dict[str, Any]],
    key: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for row in rows:
        value = str(
            row.get(key)
            or "unknown"
        )

        counts[value] = counts.get(
            value,
            0,
        ) + 1

    return counts


def _summarize(
    *,
    rows: List[Dict[str, Any]],
    total_pairs: int,
    status: str,
    old_origin: str = "",
    new_origin: str = "",
) -> Dict[str, Any]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") != "error"
    ]

    error_rows = [
        row
        for row in rows
        if row.get("status") == "error"
    ]

    n_ok = len(
        ok_rows
    )

    mean_delta = (
        sum(
            float(row.get("delta", 0))
            for row in ok_rows
        )
        / n_ok
        if n_ok
        else 0.0
    )

    # Convert [-2, 2] to [0, 1] for convenient reporting.
    normalized_soft = (
        mean_delta + 2.0
    ) / 4.0

    return {
        "status": status,
        "old_origin": old_origin,
        "new_origin": new_origin,
        "n_pairs": total_pairs,
        "n_judged": n_ok,
        "n_errors": len(error_rows),
        "mean_delta": mean_delta,
        "normalized_soft": normalized_soft,
        "improved_count": sum(
            1
            for row in ok_rows
            if int(row.get("delta", 0)) > 0
        ),
        "regressed_count": sum(
            1
            for row in ok_rows
            if int(row.get("delta", 0)) < 0
        ),
        "major_regressed_count": sum(
            1
            for row in ok_rows
            if int(row.get("delta", 0)) <= -2
        ),
        "unchanged_count": sum(
            1
            for row in ok_rows
            if int(row.get("delta", 0)) == 0
        ),
        "low_confidence_count": sum(
            1
            for row in ok_rows
            if row.get("confidence") == "low"
        ),
        "format_only_count": sum(
            1
            for row in ok_rows
            if row.get("format_only_change")
        ),
        "old_error_resolved_counts": _count_by(
            ok_rows,
            "old_error_resolved",
        ),
        "error_shift_counts": _count_by(
            ok_rows,
            "error_shift",
        ),
        "route_relation_counts": _count_by(
            ok_rows,
            "route_relation",
        ),
    }


def run_contrastive_soft(
    *,
    old_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
    out_dir: str | Path,
    workers: int = 1,
    sample_limit: int = 0,
    max_completion_tokens: int = 0,
    fail_fast: bool = False,
    env: str = "forward_math",
    old_origin: str = "",
    new_origin: str = "",
) -> Dict[str, Any]:
    """
    Pairwise contrastive soft signal.

    This function computes and writes:
      - contrastive_soft.json
      - contrastive_soft.jsonl

    The trainer may optionally use the returned summary for hcf_tiered Gate.

    old_results should be current-skill selection results.
    new_results should be candidate-skill selection results.
    """
    out_dir = Path(
        out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_by_id = _index_results(
        old_results
    )

    new_by_id = _index_results(
        new_results
    )

    sample_ids = [
        sample_id
        for sample_id in new_by_id
        if sample_id in old_by_id
    ]

    if sample_limit > 0:
        sample_ids = sample_ids[
            : sample_limit
        ]

    system_prompt = load_prompt(
        "contrastive_soft",
        env=env,
    )

    rows: List[Dict[str, Any]] = []

    if not sample_ids:
        summary = _summarize(
            rows=rows,
            total_pairs=0,
            status="skipped_no_pairs",
            old_origin=old_origin,
            new_origin=new_origin,
        )

        (
            out_dir
            / "contrastive_soft.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        _write_jsonl(
            path=out_dir / "contrastive_soft.jsonl",
            rows=rows,
        )

        return summary

    worker_count = max(
        1,
        int(
            workers
            or 1
        ),
    )

    def run_one(
        sample_id: str,
    ) -> Dict[str, Any]:
        try:
            with sample_budget_scope(sample_id):
                return judge_contrastive_pair(
                    sample_id=sample_id,
                    old_result=old_by_id[sample_id],
                    new_result=new_by_id[sample_id],
                    system_prompt=system_prompt,
                    max_completion_tokens=max_completion_tokens,
                )

        except SampleBudgetExceeded as error:
            record = _safe_error_record(
                sample_id=sample_id,
                error=error,
            )
            record["stopped"] = True
            record["stop_reason"] = "sample_operation_budget_exceeded"
            record["sample_operation_budget"] = error.to_dict()
            return record

        except BaseException as error:  # noqa: BLE001
            if fail_fast:
                raise

            return _safe_error_record(
                sample_id=sample_id,
                error=error,
            )

    if worker_count <= 1:
        rows = [
            run_one(sample_id)
            for sample_id in sample_ids
        ]

    else:
        rows_by_id: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(
            max_workers=min(
                worker_count,
                len(sample_ids),
            )
        ) as executor:
            future_to_id = {
                executor.submit(
                    run_one,
                    sample_id,
                ): sample_id
                for sample_id in sample_ids
            }

            for future in as_completed(
                future_to_id
            ):
                sample_id = future_to_id[
                    future
                ]

                rows_by_id[sample_id] = future.result()

        rows = [
            rows_by_id[sample_id]
            for sample_id in sample_ids
            if sample_id in rows_by_id
        ]

    status = (
        "complete"
        if all(
            row.get("status") != "error"
            for row in rows
        )
        else "partial"
    )

    summary = _summarize(
        rows=rows,
        total_pairs=len(sample_ids),
        status=status,
        old_origin=old_origin,
        new_origin=new_origin,
    )

    (
        out_dir
        / "contrastive_soft.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        path=out_dir / "contrastive_soft.jsonl",
        rows=rows,
    )

    return summary
