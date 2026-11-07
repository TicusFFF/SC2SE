from __future__ import annotations

from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, List

from S5_bridge.a0_rollout_prompt import (
    build_a0_messages,
    load_rollout_system,
)
from S5_bridge.data_sanitizer import (
    assert_blind_item,
)
from S5_bridge.forward_evidence import (
    ForwardRolloutEvidence,
    evidence_to_conversation,
    evidence_to_skillopt_result,
)
from S5_bridge.forward_runner import (
    ForwardSkillRunner,
)
from skillopt.evaluation.strict_validation import (
    FORWARD_PASS_RATE,
    REAL_ACC,
    normalize_strict_validation_metric,
    selection_eval_protocol,
)
from skillopt.optimizer.structured_skills import (
    parse_skill_library_text,
    skill_behavior_hash,
)


try:
    from mod_C0_compress import (
        trace_scope,
    )
except ImportError:
    # Keep ForwardMath usable as a standalone B0 environment without C0.
    def trace_scope(*, sample_id, trace_path):
        return nullcontext()

try:
    from utils.sample_budget import (
        SampleBudgetExceeded,
        check_sample_budget,
        sample_budget_scope,
    )
except ImportError:
    class SampleBudgetExceeded(BaseException):
        pass

    def check_sample_budget():
        return None

    def sample_budget_scope(sample_id):
        return nullcontext()


RunnerFactory = Callable[
    [],
    ForwardSkillRunner,
]

SELECTION_EVAL_PROTOCOL = selection_eval_protocol(
    REAL_ACC
)
STUDENT_RESPONSE_CACHE_SCHEMA_VERSION = 1


def _progress_phase(
    out_dir: Path,
) -> str:
    if out_dir.name == "rollout" and out_dir.parent.name:
        return f"{out_dir.parent.name}/rollout"

    return out_dir.name or "rollout"


def is_initial_bootstrap_rollout_phase(
    out_dir: str | Path,
) -> bool:
    """Return True only for the first training rollout.

    This is the S5 bootstrap stage: use the pre-generated shared
    initial answer as the starting answer, then let Forward produce
    verification/repair evidence for SkillOpt. Candidate selection
    evals and later rollouts still use A0.

    Supports both default accumulation=1:
        .../steps/step_0001/rollout
    and accumulation>1:
        .../steps/step_0001/batch_0/rollout
    """

    path = Path(out_dir)

    if path.name != "rollout":
        return False

    return any(
        parent.name == "step_0001"
        and parent.parent.name == "steps"
        for parent in path.parents
    )


def uses_shared_initial_response_phase(
    out_dir: str | Path,
    *,
    bootstrap_shared_initial_response: bool = True,
) -> bool:
    return (
        bootstrap_shared_initial_response
        and is_initial_bootstrap_rollout_phase(
            Path(out_dir)
        )
    )


def initial_response_source_label(
    out_dir: str | Path,
    *,
    bootstrap_shared_initial_response: bool = True,
) -> str:
    path = Path(out_dir)

    if (
        bootstrap_shared_initial_response
        and is_initial_bootstrap_rollout_phase(path)
    ):
        return "shared_initial_response_bootstrap"

    return "skill_generated"


def is_student_only_evaluation_phase(
    out_dir: str | Path,
) -> bool:
    return Path(out_dir).name in {
        "selection_eval_baseline",
        "selection_eval",
        "test_eval_baseline",
        "test_eval",
    }


def _stable_json_hash(
    value: Dict[str, Any],
) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def _selection_skill_hash(
    skill_content: str,
) -> str:
    library = parse_skill_library_text(
        str(skill_content or "")
    )
    return skill_behavior_hash(library)


def _selection_request_identity(
    a0_messages: List[Dict[str, str]],
    selection_metric: str = REAL_ACC,
) -> Dict[str, Any]:
    metric = normalize_strict_validation_metric(
        selection_metric
    )
    return {
        "schema_version": (
            STUDENT_RESPONSE_CACHE_SCHEMA_VERSION
        ),
        "selection_eval_protocol": (
            selection_eval_protocol(metric)
        ),
        "messages": deepcopy(
            a0_messages
        ),
    }


def _selection_metric_from_row(
    row: Dict[str, Any],
) -> str:
    raw_metric = str(
        row.get("selection_metric")
        or ""
    ).strip().lower()

    # Read-only compatibility for results written before the public
    # metric name became ``real_acc``.
    if raw_metric == "strict_acc":
        return REAL_ACC

    return normalize_strict_validation_metric(
        raw_metric
    )


def _build_student_response_reuse_map(
    rows: Iterable[Dict[str, Any]] | None,
) -> Dict[str, Dict[str, Any]] | None:
    if rows is None:
        return None

    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                "Each student response reuse row must be a dict."
            )

        sample_id = str(
            row.get("id")
            or ""
        )

        if not sample_id:
            raise ValueError(
                "Student response reuse row is missing id."
            )

        if sample_id in result:
            raise ValueError(
                "Student response reuse rows contain duplicate "
                f"sample id {sample_id!r}."
            )

        try:
            metric = _selection_metric_from_row(
                row
            )
        except ValueError as error:
            raise ValueError(
                "Student response reuse row has an incompatible "
                f"selection metric for sample {sample_id!r}."
            ) from error

        if row.get("selection_eval_protocol") != (
            selection_eval_protocol(metric)
        ):
            raise ValueError(
                "Student response reuse row has an incompatible "
                f"selection protocol for sample {sample_id!r}."
            )

        if metric == REAL_ACC:
            if (
                not isinstance(
                    row.get(
                        "real_acc_correct",
                        row.get("strict_acc_correct"),
                    ),
                    bool,
                )
                or not isinstance(
                    row.get(
                        "real_acc_eval_row",
                        row.get("strict_acc_eval_row"),
                    ),
                    dict,
                )
            ):
                raise ValueError(
                    "Student response reuse row has not completed "
                    f"real ACC judging for sample {sample_id!r}."
                )
        elif (
            not isinstance(
                row.get("forward_pass_rate_correct"),
                bool,
            )
            or len(
                row.get("forward_steps")
                if isinstance(
                    row.get("forward_steps"),
                    list,
                )
                else []
            ) != 1
            or str(
                row.get(
                    "verification_unavailable_reason",
                    "",
                )
                or ""
            ).strip()
        ):
            raise ValueError(
                "Student response reuse row has not completed "
                "one available Forward verification for sample "
                f"{sample_id!r}."
            )

        if not str(
            row.get("selection_skill_hash")
            or ""
        ).strip():
            raise ValueError(
                "Student response reuse row is missing its Skill hash "
                f"entry for sample {sample_id!r}."
            )

        if not str(
            row.get("selection_request_hash")
            or ""
        ).strip():
            raise ValueError(
                "Student response reuse row is missing its request hash "
                f"for sample {sample_id!r}."
            )

        result[sample_id] = deepcopy(
            row
        )

    return result


def _resolve_student_response_override(
    *,
    sample_id: str,
    skill_content: str,
    a0_messages: List[Dict[str, str]],
    reuse_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        row = reuse_map[sample_id]
    except KeyError as error:
        raise KeyError(
            "Student response reuse results are missing "
            f"sample id {sample_id!r}."
        ) from error

    metric = _selection_metric_from_row(
        row
    )
    expected_identity = (
        _selection_request_identity(
            a0_messages,
            metric,
        )
    )
    expected_request_hash = (
        _stable_json_hash(
            expected_identity
        )
    )
    expected_skill_hash = (
        _selection_skill_hash(
            skill_content
        )
    )

    if (
        row.get("selection_eval_protocol")
        != selection_eval_protocol(metric)
        or row.get("selection_skill_hash")
        != expected_skill_hash
        or row.get("selection_request_hash")
        != expected_request_hash
    ):
        raise ValueError(
            "Student response reuse identity does not match the "
            f"current Skill and prompt for sample {sample_id!r}."
        )

    response = str(
        row.get("initial_response")
        or ""
    ).strip()
    pred = str(
        row.get("initial_answer")
        or ""
    ).strip()

    if not response or not pred:
        raise ValueError(
            "Student response reuse entry contains an empty "
            f"response or answer for sample {sample_id!r}."
        )

    return {
        "response": response,
        "pred": pred,
        "answer_extraction_fallback": str(
            row.get(
                "answer_extraction_fallback"
            )
            or ""
        ),
    }


def _print_s5_progress(
    *,
    phase: str,
    done: int,
    total: int,
    sample_id: str,
) -> None:
    print(
        f"[S5 progress] phase={phase} done={done}/{total} sample={sample_id}",
        flush=True,
    )


def _as_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    text = str(value or "").strip().lower()
    return text in {
        "1",
        "true",
        "yes",
        "pass",
        "passed",
        "ok",
    }


def _result_initial_pass(
    result: Dict[str, Any],
) -> bool:
    if "forward_initial_pass" in result:
        return _as_bool(result.get("forward_initial_pass"))

    return _as_bool(result.get("hard"))


def _result_final_pass(
    result: Dict[str, Any],
) -> bool:
    if "forward_final_pass" in result:
        return _as_bool(result.get("forward_final_pass"))

    steps = result.get("forward_steps") or []
    if not isinstance(steps, list):
        return False

    return any(
        _as_bool(step.get("r_f"))
        for step in steps
        if isinstance(step, dict)
    )


def _result_repair_attempted(
    result: Dict[str, Any],
) -> bool:
    if "forward_repair_attempted" in result:
        return _as_bool(result.get("forward_repair_attempted"))

    if _result_initial_pass(result):
        return False

    steps = result.get("forward_steps") or []
    if not isinstance(steps, list):
        return False

    return any(
        bool(step.get("y_prime"))
        for step in steps
        if isinstance(step, dict)
    )


def _metrics_dataset_root(
    out_dir: Path,
) -> Path:
    for parent in (
        out_dir,
        *out_dir.parents,
    ):
        if parent.name == "steps":
            return parent.parent

    return out_dir.parent


def _metrics_phase(
    out_dir: Path,
) -> str:
    root = _metrics_dataset_root(out_dir)
    try:
        phase = out_dir.relative_to(root).as_posix()
    except ValueError:
        phase = _progress_phase(out_dir)

    if phase == "selection_eval_baseline":
        return "baseline"

    return phase


def _pct(
    numerator: int,
    denominator: int,
) -> str:
    if denominator <= 0:
        return "0.0%"

    return f"{100.0 * numerator / denominator:.1f}%"


def _summarize_s5_metrics(
    *,
    phase: str,
    results: List[Dict[str, Any]],
    status: str,
    skill_count: int,
) -> Dict[str, Any]:
    total = len(results)
    initial_correct = sum(
        1
        for result in results
        if _result_initial_pass(result)
    )
    final_correct = sum(
        1
        for result in results
        if _result_final_pass(result)
    )
    repair_attempted = sum(
        1
        for result in results
        if _result_repair_attempted(result)
    )
    repair_accepted = sum(
        1
        for result in results
        if (
            not _result_initial_pass(result)
            and _result_final_pass(result)
        )
    )
    trajectory_counts = Counter(
        str(result.get("trajectory_type") or "unknown")
        for result in results
    )
    source_counts = Counter(
        str(result.get("initial_response_source") or "unknown")
        for result in results
    )
    a0_called = sum(
        1
        for result in results
        if _as_bool(result.get("a0_called"))
    )

    return {
        "phase": phase,
        "status": status,
        "n": total,
        "skill_count": int(
            skill_count
        ),
        "initial_correct": initial_correct,
        "initial_acc": (
            initial_correct / total
            if total
            else 0.0
        ),
        "forward_final_correct": final_correct,
        "forward_final_acc": (
            final_correct / total
            if total
            else 0.0
        ),
        "forward_gain": (
            (final_correct - initial_correct) / total
            if total
            else 0.0
        ),
        "repair_attempted": repair_attempted,
        "repair_accepted": repair_accepted,
        "trajectory_counts": dict(trajectory_counts),
        "initial_response_source_counts": dict(source_counts),
        "a0_called": a0_called,
    }


def _count_skill_entries(
    skill_content: str,
) -> int:
    library = parse_skill_library_text(
        str(skill_content or "")
    )
    return len(library["skills"])


_STEP_JUDGE_MODULE = None


def _load_step_judge_module():
    global _STEP_JUDGE_MODULE

    if _STEP_JUDGE_MODULE is not None:
        return _STEP_JUDGE_MODULE

    for parent in Path(__file__).resolve().parents:
        module_path = (
            parent
            / "0-2_step_judge"
            / "eval_hard_test_outputs.py"
        )

        if module_path.is_file():
            judge_dir = str(module_path.parent)
            if judge_dir not in sys.path:
                sys.path.insert(
                    0,
                    judge_dir,
                )

            spec = importlib.util.spec_from_file_location(
                "s5_step_judge_eval",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Cannot load 0-2_step_judge from {module_path}"
                )

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _STEP_JUDGE_MODULE = module
            return module

    raise FileNotFoundError(
        "Cannot find 0-2_step_judge/eval_hard_test_outputs.py "
        "from the S5 rollout module path."
    )


def _result_to_judge_log(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    initial_answer = str(
        result.get("initial_answer")
        or ""
    )
    final_answer = str(
        result.get("final_answer")
        or initial_answer
    )

    log = {
        "id": str(result.get("id") or ""),
        "initial_result": initial_answer,
        "final_result": final_answer,
    }

    initial_response = result.get(
        "initial_response"
    )
    if str(initial_response or "").strip():
        log["initial_response"] = initial_response

    final_response = result.get(
        "final_response"
    )
    if str(final_response or "").strip():
        log["final_response"] = final_response

    return log


def _attach_cached_selection_acc_metrics(
    *,
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> bool:
    """Reuse strict selection rows instead of judging the same answer twice."""
    if not results:
        return False

    rows: List[Dict[str, Any]] = []

    for result in results:
        try:
            metric = _selection_metric_from_row(
                result
            )
        except ValueError:
            return False

        if (
            result.get("selection_eval_protocol")
            != selection_eval_protocol(REAL_ACC)
            or metric != REAL_ACC
            or not isinstance(
                result.get(
                    "real_acc_correct",
                    result.get("strict_acc_correct"),
                ),
                bool,
            )
            or not isinstance(
                result.get(
                    "real_acc_eval_row",
                    result.get("strict_acc_eval_row"),
                ),
                dict,
            )
        ):
            return False

        row = dict(
            result.get(
                "real_acc_eval_row",
                result.get("strict_acc_eval_row"),
            )
        )

        if str(row.get("id") or "") != str(
            result.get("id") or ""
        ):
            return False

        rows.append(row)

    correct_ids = [
        str(result.get("id") or "")
        for result in results
        if result.get(
            "real_acc_correct",
            result.get("strict_acc_correct"),
        )
    ]
    real_n = len(results)
    real_correct = len(correct_ids)

    summary["real_acc_status"] = "ok"
    summary["real_judge"] = (
        "reused_selection_strict_acc"
    )
    summary["real_n"] = real_n
    summary["real_initial_correct"] = (
        real_correct
    )
    summary["real_final_correct"] = (
        real_correct
    )
    summary["real_initial_acc"] = (
        real_correct / real_n
        if real_n
        else 0.0
    )
    summary["real_final_acc"] = (
        real_correct / real_n
        if real_n
        else 0.0
    )
    summary["real_improved_ids"] = []
    summary["real_regressed_ids"] = []
    summary["real_correct_ids"] = (
        correct_ids
    )
    summary["real_acc_reused"] = True
    summary["_real_eval_rows"] = rows

    return True


def _attach_real_acc_metrics(
    *,
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    judge_data_path: str,
    judge_dataset_key: str,
    judge_workers: int,
) -> None:
    if _attach_cached_selection_acc_metrics(
        summary=summary,
        results=results,
    ):
        return

    if not judge_data_path:
        summary["real_acc_status"] = "skipped_no_dataset_path"
        return

    try:
        module = _load_step_judge_module()
        logs = [
            _result_to_judge_log(result)
            for result in results
        ]
        real_summary = module.make_eval_summary_from_logs(
            data_path=judge_data_path,
            logs=logs,
            dataset_key=judge_dataset_key,
            method=str(summary.get("phase") or "s5"),
            model="",
            workers=max(
                1,
                int(judge_workers or 1),
            ),
        )
    except Exception as error:  # noqa: BLE001
        summary["real_acc_status"] = "error"
        summary["real_acc_error"] = str(error)
        summary["real_acc_error_type"] = type(error).__name__
        return

    real_n = int(
        real_summary.get("n")
        or 0
    )
    real_initial_correct = int(
        real_summary.get("initial_correct")
        or 0
    )
    real_final_correct = int(
        real_summary.get("final_correct")
        or 0
    )

    summary["real_acc_status"] = "ok"
    summary["real_judge"] = real_summary.get(
        "judge"
    )
    summary["real_n"] = real_n
    summary["real_initial_correct"] = real_initial_correct
    summary["real_final_correct"] = real_final_correct
    summary["real_initial_acc"] = (
        real_initial_correct / real_n
        if real_n
        else 0.0
    )
    summary["real_final_acc"] = (
        real_final_correct / real_n
        if real_n
        else 0.0
    )
    summary["real_improved_ids"] = real_summary.get(
        "improved_ids",
        [],
    )
    summary["real_regressed_ids"] = real_summary.get(
        "regressed_ids",
        [],
    )
    summary["_real_eval_rows"] = [
        {
            "id": str(row.get("id") or ""),
            "initial_correct": bool(
                row.get("initial_correct")
            ),
            "final_correct": bool(
                row.get("final_correct")
            ),
            "initial_empty_response": bool(
                row.get("initial_empty_response")
            ),
            "final_empty_response": bool(
                row.get("final_empty_response")
            ),
            "empty_response": bool(
                row.get("empty_response")
            ),
        }
        for row in real_summary.get("rows", [])
        if isinstance(row, dict)
    ]


def _attach_selection_strict_acc(
    *,
    results: List[Dict[str, Any]],
    judge_data_path: str,
    judge_dataset_key: str,
    judge_workers: int,
) -> None:
    if not results:
        return

    summary: Dict[str, Any] = {
        "phase": "selection_strict_acc",
    }

    _attach_real_acc_metrics(
        summary=summary,
        results=results,
        judge_data_path=judge_data_path,
        judge_dataset_key=(
            judge_dataset_key
        ),
        judge_workers=judge_workers,
    )

    if summary.get("real_acc_status") != "ok":
        raise RuntimeError(
            "Strict ACC evaluation failed: "
            + str(
                summary.get(
                    "real_acc_error",
                    summary.get(
                        "real_acc_status",
                        "unknown error",
                    ),
                )
            )
        )

    rows = summary.get(
        "_real_eval_rows"
    )

    if not isinstance(rows, list):
        raise RuntimeError(
            "Strict ACC evaluation returned no per-sample rows."
        )

    rows_by_id: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(
                "Strict ACC row must be a dict."
            )

        sample_id = str(
            row.get("id")
            or ""
        )

        if not sample_id or sample_id in rows_by_id:
            raise RuntimeError(
                "Strict ACC rows contain a missing or duplicate "
                "sample id."
            )

        rows_by_id[sample_id] = row

    for result in results:
        sample_id = str(
            result.get("id")
            or ""
        )
        row = rows_by_id.get(
            sample_id
        )

        if row is None:
            raise RuntimeError(
                "Strict ACC evaluation is missing "
                f"sample {sample_id!r}."
            )

        strict_correct = bool(
            row.get("final_correct")
        )

        result["selection_eval_protocol"] = (
            selection_eval_protocol(
                REAL_ACC
            )
        )
        result["selection_metric"] = (
            REAL_ACC
        )
        result["real_acc_correct"] = (
            strict_correct
        )
        result["real_acc_eval_row"] = dict(
            row
        )
        # Preserve the established field names for old reports while
        # exposing the public metric name used by new strict screening.
        result["strict_acc_correct"] = (
            strict_correct
        )
        result["strict_acc_eval_row"] = dict(
            row
        )

        # Keep the established score/cache/gate plumbing. In Student-only
        # evaluation phases hard and soft both mean strict correctness.
        result["hard"] = float(
            strict_correct
        )
        result["soft"] = float(
            strict_correct
        )
        result["trajectory_type"] = (
            "selection_student_only"
        )
        result["fail_reason"] = (
            ""
            if strict_correct
            else (
                "Strict answer judge marked the "
                "Student answer incorrect."
            )
        )


def _attach_selection_forward_pass_rate(
    *,
    results: List[Dict[str, Any]],
) -> None:
    protocol = selection_eval_protocol(
        FORWARD_PASS_RATE
    )

    for result in results:
        steps = result.get(
            "forward_steps"
        )

        if (
            not isinstance(steps, list)
            or len(steps) != 1
            or not isinstance(steps[0], dict)
        ):
            raise RuntimeError(
                "Forward-pass-rate selection must contain "
                "exactly one complete Forward step."
            )

        step = steps[0]
        if any(
            step.get(key)
            for key in (
                "F_f",
                "y_prime",
                "a_prime",
            )
        ) or bool(
            result.get("forward_repair_attempted")
        ) or bool(
            result.get("forward_repair_accepted")
        ):
            raise RuntimeError(
                "Forward-pass-rate selection must stop before "
                "Forward correction."
            )

        if (
            str(result.get("final_response") or "")
            != str(result.get("initial_response") or "")
            or str(result.get("final_answer") or "")
            != str(result.get("initial_answer") or "")
        ):
            raise RuntimeError(
                "Forward-pass-rate selection must preserve the "
                "original Student answer."
            )

        unavailable_reason = str(
            result.get(
                "verification_unavailable_reason",
                "",
            )
            or ""
        ).strip()

        # A known duplicate-loop or tool-flow-limit stop is not evidence
        # that the answer is wrong. Keep it in the denominator without
        # awarding a pass; preserve its unavailable status for reporting
        # and reuse checks.
        if unavailable_reason and unavailable_reason not in (
            "verification_unavailable_duplicate_loop",
            "verification_unavailable_tool_flow_loop_exhausted",
        ):
            raise RuntimeError(
                "Forward-pass-rate selection cannot score an "
                "unavailable Forward verification: "
                + unavailable_reason
            )

        forward_pass = bool(
            result.get("forward_initial_pass")
        )
        if bool(step.get("r_f")) != forward_pass:
            raise RuntimeError(
                "Forward-pass-rate result disagrees with its "
                "complete Forward step."
            )

        if unavailable_reason and (
            forward_pass
            or bool(result.get("forward_final_pass"))
        ):
            raise RuntimeError(
                "Unavailable Forward verification must not be "
                "marked passed."
            )

        result["selection_eval_protocol"] = (
            protocol
        )
        result["selection_metric"] = (
            FORWARD_PASS_RATE
        )
        result["forward_pass_rate_correct"] = (
            forward_pass
        )
        result["hard"] = float(
            forward_pass
        )
        result["soft"] = float(
            forward_pass
        )
        result["trajectory_type"] = (
            "verifier_error"
            if unavailable_reason
            else "selection_student_first_forward_only"
        )
        result["fail_reason"] = (
            "Forward verification unavailable: " + unavailable_reason
            if unavailable_reason
            else (
                ""
                if forward_pass
                else "Forward verification marked the Student answer incorrect."
            )
        )


def _attach_selection_metric(
    *,
    results: List[Dict[str, Any]],
    strict_validation_metric: str,
    judge_data_path: str,
    judge_dataset_key: str,
    judge_workers: int,
) -> None:
    metric = normalize_strict_validation_metric(
        strict_validation_metric
    )

    if metric == FORWARD_PASS_RATE:
        _attach_selection_forward_pass_rate(
            results=results,
        )
        return

    _attach_selection_strict_acc(
        results=results,
        judge_data_path=judge_data_path,
        judge_dataset_key=judge_dataset_key,
        judge_workers=judge_workers,
    )


def _load_metrics_history(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []

    history: List[Dict[str, Any]] = []
    for line in path.read_text(
        encoding="utf-8",
    ).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            history.append(value)

    return history


def _store_metrics_history(
    path: Path,
    history: List[Dict[str, Any]],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            for item in history
        ),
        encoding="utf-8",
    )


def _write_jsonl_rows(
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


def _metrics_table_lines(
    history: List[Dict[str, Any]],
) -> List[str]:
    rows: List[Dict[str, str]] = []
    baseline_skill_count = None

    for item in history:
        n = int(item.get("n") or 0)
        skill_count = int(
            item.get("skill_count")
            or 0
        )
        if baseline_skill_count is None:
            baseline_skill_count = skill_count

        skill_delta = (
            skill_count
            - baseline_skill_count
        )
        real_n = int(
            item.get("real_n")
            or n
        )
        initial_correct = int(
            item.get("initial_correct")
            or 0
        )
        final_correct = int(
            item.get("forward_final_correct")
            or 0
        )
        repair_accepted = int(
            item.get("repair_accepted")
            or 0
        )
        repair_attempted = int(
            item.get("repair_attempted")
            or 0
        )
        counts = item.get("trajectory_counts") or {}
        gain = final_correct - initial_correct
        real_ok = item.get("real_acc_status") == "ok"
        real_initial_correct = int(
            item.get("real_initial_correct")
            or 0
        )
        real_final_correct = int(
            item.get("real_final_correct")
            or 0
        )

        rows.append(
            {
                "phase": str(item.get("phase", "-")),
                "status": str(item.get("status", "-")),
                "n": str(n),
                "skill_n": (
                    f"{skill_count}({skill_delta:+d})"
                ),
                "real_init_acc": (
                    f"{real_initial_correct}/{real_n}="
                    f"{_pct(real_initial_correct, real_n)}"
                    if real_ok
                    else "n/a"
                ),
                "real_final_acc": (
                    f"{real_final_correct}/{real_n}="
                    f"{_pct(real_final_correct, real_n)}"
                    if real_ok
                    else "n/a"
                ),
                "init_fwd_pass": (
                    f"{initial_correct}/{n}="
                    f"{_pct(initial_correct, n)}"
                ),
                "final_fwd_pass": (
                    f"{final_correct}/{n}="
                    f"{_pct(final_correct, n)}"
                ),
                "fwd_gain": f"{gain:+d}={_pct(gain, n)}",
                "repair_ok": (
                    f"{repair_accepted}/"
                    f"{repair_attempted}"
                ),
                "passed": str(
                    int(counts.get("passed") or 0)
                ),
                "failed_repaired": str(
                    int(counts.get("failed_repaired") or 0)
                ),
                "failed": str(
                    int(counts.get("failed") or 0)
                ),
                "verifier_error": str(
                    int(counts.get("verifier_error") or 0)
                ),
            }
        )

    columns = [
        ("phase", "phase", "left"),
        ("status", "status", "left"),
        ("n", "n", "right"),
        ("skill_n", "skill_n(delta_vs_baseline)", "right"),
        ("real_init_acc", "real_init_acc", "right"),
        ("real_final_acc", "real_final_acc", "right"),
        ("init_fwd_pass", "init_fwd_pass", "right"),
        ("final_fwd_pass", "final_fwd_pass", "right"),
        ("fwd_gain", "fwd_gain", "right"),
        ("repair_ok", "repair_ok", "right"),
        ("passed", "passed", "right"),
        ("failed_repaired", "failed_repaired", "right"),
        ("failed", "failed", "right"),
        ("verifier_error", "verifier_error", "right"),
    ]

    widths = {
        key: max(
            len(header),
            *(
                len(row[key])
                for row in rows
            ),
        )
        for key, header, _align in columns
    }

    def format_cell(
        value: str,
        *,
        key: str,
        align: str,
    ) -> str:
        width = widths[key]
        if align == "left":
            return value.ljust(width)
        return value.rjust(width)

    def format_row(
        row: Dict[str, str],
    ) -> str:
        return "  ".join(
            format_cell(
                row[key],
                key=key,
                align=align,
            )
            for key, _header, align in columns
        )

    latest = history[-1]
    context_model = str(
        latest.get("model")
        or "-"
    )
    context_dataset = str(
        latest.get("dataset_key")
        or "-"
    )
    context_step = str(
        latest.get("phase")
        or "-"
    )
    context_status = str(
        latest.get("status")
        or "-"
    )
    context_n = str(
        latest.get("n")
        or "-"
    )

    lines = [
        (
            "[S5 metrics] context "
            f"model={context_model} "
            f"dataset={context_dataset} "
            f"step={context_step} "
            f"status={context_status} "
            f"n={context_n}"
        ),
        (
            "[S5 metrics] real_*_acc is computed by 0-2_step_judge "
            "when real-ACC judging is enabled; selection_metric "
            "records whether strict screening used real_acc or "
            "forward_pass_rate."
        ),
        (
            "[S5 metrics] *_fwd_pass is Forward verifier pass rate; "
            "init_fwd_pass uses the first Forward check and "
            "final_fwd_pass uses any passed Forward check."
        ),
        "[S5 metrics] "
        + format_row(
            {
                key: header
                for key, header, _align in columns
            }
        ),
        "[S5 metrics] "
        + "  ".join(
            "-" * widths[key]
            for key, _header, _align in columns
        ),
    ]

    for row in rows:
        lines.append(
            "[S5 metrics] "
            + format_row(row)
        )

    errors = [
        str(item.get("real_acc_error"))
        for item in history
        if item.get("real_acc_status") == "error"
    ]
    if errors:
        lines.append(
            "[S5 metrics] real_acc warning: "
            + errors[-1]
        )

    return lines


def _emit_s5_metrics(
    *,
    out_dir: Path,
    results: List[Dict[str, Any]],
    status: str,
    skill_count: int,
    judge_data_path: str = "",
    judge_dataset_key: str = "",
    metrics_model: str = "",
    judge_workers: int = 1,
) -> None:
    if not results:
        return

    phase = _metrics_phase(out_dir)
    summary = _summarize_s5_metrics(
        phase=phase,
        results=results,
        status=status,
        skill_count=skill_count,
    )
    summary["dataset_key"] = str(
        judge_dataset_key
        or "-"
    )
    summary["model"] = str(
        metrics_model
        or "-"
    )

    if status == "complete":
        _attach_real_acc_metrics(
            summary=summary,
            results=results,
            judge_data_path=judge_data_path,
            judge_dataset_key=judge_dataset_key,
            judge_workers=judge_workers,
        )

    real_eval_rows = summary.pop(
        "_real_eval_rows",
        None,
    )
    if real_eval_rows is not None:
        _write_jsonl_rows(
            out_dir / "forward_real_eval_rows.jsonl",
            real_eval_rows,
        )

    (
        out_dir
        / "forward_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    root = _metrics_dataset_root(out_dir)
    history_path = root / "s5_forward_metrics.jsonl"

    if status == "complete":
        history = [
            item
            for item in _load_metrics_history(history_path)
            if not (
                item.get("phase") == phase
                and item.get("status") == status
            )
        ]
        history.append(summary)
        _store_metrics_history(
            history_path,
            history,
        )
    else:
        history = [summary]

    for line in _metrics_table_lines(history):
        print(
            line,
            flush=True,
        )


def _write_results_and_metrics(
    *,
    out_dir: Path,
    results: List[Dict[str, Any]],
    status: str,
    skill_count: int,
    judge_data_path: str = "",
    judge_dataset_key: str = "",
    metrics_model: str = "",
    judge_workers: int = 1,
) -> None:
    if not results:
        return

    file_name = (
        "results.jsonl"
        if status == "complete"
        else "results.partial.jsonl"
    )

    (
        out_dir
        / file_name
    ).write_text(
        "\n".join(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            for result in results
        ),
        encoding="utf-8",
    )

    _emit_s5_metrics(
        out_dir=out_dir,
        results=results,
        status=status,
        skill_count=skill_count,
        judge_data_path=judge_data_path,
        judge_dataset_key=judge_dataset_key,
        metrics_model=metrics_model,
        judge_workers=judge_workers,
    )


def _run_forward_math_item(
    *,
    item: Dict[str, Any],
    skill_content: str,
    out_dir: Path,
    runner: ForwardSkillRunner,
    rollout_system_template: str,
    shared_initial_response_map: Dict[str, Dict[str, Any]] | None,
    bootstrap_shared_initial_response: bool = True,
    student_only_evaluation: bool = False,
    student_forward_verification_only: bool = False,
    selection_metric: str | None = None,
    student_response_reuse_map: (
        Dict[str, Dict[str, Any]] | None
    ) = None,
) -> Dict[str, Any]:
    assert_blind_item(item)

    sample_id = str(
        item["id"]
    )

    problem = (
        item.get("q")
        or item.get("query")
        or item.get("problem")
    )

    if not problem:
        raise KeyError(
            f"Missing problem text for sample {sample_id}"
        )

    pred_dir = (
        out_dir
        / "predictions"
        / sample_id
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build the A0 prompt even when this phase uses a shared initial
    # response. The prompt is passed to A0 only when no override is used.
    a0_messages = build_a0_messages(
        problem=problem,
        skill_content=skill_content,
        rollout_system_template=(
            rollout_system_template
        ),
    )

    initial_response_override = None
    student_response_reused = False
    answer_extraction_fallback = ""
    source_label = initial_response_source_label(
        out_dir,
        bootstrap_shared_initial_response=(
            bootstrap_shared_initial_response
        ),
    )

    # The public bootstrap answer has priority over any saved selection
    # response.  It is the original S5 first-round evidence source.
    if shared_initial_response_map is not None:
        try:
            initial_response_override = (
                shared_initial_response_map[
                    sample_id
                ]
            )
        except KeyError as error:
            raise KeyError(
                "Shared initial response file is missing "
                f"sample id {sample_id!r}."
            ) from error

        if source_label == "skill_generated":
            source_label = "shared_initial_response"

    elif student_response_reuse_map is not None:
        if (
            student_only_evaluation
            or student_forward_verification_only
        ):
            raise ValueError(
                "Student evaluation cannot consume its own "
                "selection response reuse results."
            )

        initial_response_override = (
            _resolve_student_response_override(
                sample_id=sample_id,
                skill_content=skill_content,
                a0_messages=a0_messages,
                reuse_map=(
                    student_response_reuse_map
                ),
            )
        )
        student_response_reused = True
        answer_extraction_fallback = str(
            initial_response_override.get(
                "answer_extraction_fallback"
            )
            or ""
        )
        source_label = (
            "selection_eval_reuse"
        )

    elif student_only_evaluation:
        source_label = (
            "skill_generated_student_only"
        )
    elif student_forward_verification_only:
        source_label = (
            "skill_generated_student_first_forward_only"
        )

    a0_called = initial_response_override is None

    budget_stop = None
    budget_stop_reason = ""
    try:
        with sample_budget_scope(sample_id), trace_scope(
            sample_id=sample_id,
            trace_path=(
                pred_dir
                / "context_compression.jsonl"
            ),
        ):
            evidence = runner.run_one_with_messages(
                item=item,
                a0_messages=a0_messages,
                skill_content=skill_content,
                initial_response_override=(
                    initial_response_override
                ),
                student_only=(
                    student_only_evaluation
                ),
                student_forward_verification_only=(
                    student_forward_verification_only
                ),
            )

            if (
                student_only_evaluation
                or student_forward_verification_only
            ):
                answer_extraction_fallback = str(
                    getattr(
                        runner,
                        "last_answer_extraction_fallback",
                        "",
                    )
                    or ""
                )
            check_sample_budget()
    except SampleBudgetExceeded as error:
        initial = initial_response_override or {}
        initial_response = str(
            initial.get("response")
            or getattr(runner, "last_initial_response", "")
        )
        initial_answer = str(
            initial.get("pred")
            or getattr(runner, "last_initial_answer", "")
        )
        evidence = ForwardRolloutEvidence(
            sample_id=sample_id,
            q=str(problem),
            skill_version=runner.skill_version,
            y=initial_response,
            a=initial_answer,
            steps=[],
            final_response=initial_response,
            final_answer=initial_answer,
        )
        budget_stop = error.to_dict()
        budget_stop_reason = error.stop_reason

    result = evidence_to_skillopt_result(
        evidence
    )

    if budget_stop is not None:
        result["stopped"] = True
        result["stop_reason"] = budget_stop_reason
        result["sample_operation_budget"] = budget_stop

    result["initial_response_source"] = source_label
    result["a0_called"] = a0_called
    result["a0_call_count"] = (
        1 if a0_called else 0
    )
    result["bootstrap_shared_initial_response"] = (
        source_label
        == "shared_initial_response_bootstrap"
    )
    result["student_response_reused"] = (
        student_response_reused
    )
    result["answer_extraction_fallback"] = (
        answer_extraction_fallback
    )

    request_metric = selection_metric
    if (
        request_metric is None
        and student_response_reuse_map is not None
    ):
        request_metric = _selection_metric_from_row(
            student_response_reuse_map[sample_id]
        )
    if request_metric is None:
        request_metric = (
            FORWARD_PASS_RATE
            if student_forward_verification_only
            else REAL_ACC
        )

    request_identity = (
        _selection_request_identity(
            a0_messages,
            request_metric,
        )
    )
    result["selection_skill_hash"] = (
        _selection_skill_hash(
            skill_content
        )
    )
    result["selection_request_hash"] = (
        _stable_json_hash(
            request_identity
        )
    )

    if (
        student_only_evaluation
        or student_forward_verification_only
    ):
        evaluation_metric = (
            normalize_strict_validation_metric(
                selection_metric
                if selection_metric is not None
                else (
                    FORWARD_PASS_RATE
                    if student_forward_verification_only
                    else REAL_ACC
                )
            )
        )
        result["selection_eval_protocol"] = (
            selection_eval_protocol(
                evaluation_metric
            )
        )
        result["selection_metric"] = (
            evaluation_metric
        )
        result["trajectory_type"] = (
            (
                "selection_student_only"
                if student_only_evaluation
                else "selection_student_first_forward_only"
            )
        )

    result["prediction_dir"] = str(
        pred_dir
    )

    if not (
        source_label
        in {
            "shared_initial_response_bootstrap",
            "shared_initial_response",
        }
    ):
        target_system_prompt = a0_messages[0]["content"]
        target_user_prompt = a0_messages[1]["content"]
    else:
        # The shared first response was generated outside S5/A0.
        # In the original Forward first-response path, the model sees
        # the problem as the user message; no S5 skill-conditioned
        # system prompt is used.
        #
        # Keep target_* files aligned with the actual initial-response
        # source so SkillOpt reflection does not attribute bootstrap
        # behavior to A0.
        target_system_prompt = ""
        target_user_prompt = str(problem)

    (
        pred_dir
        / "target_system_prompt.txt"
    ).write_text(
        target_system_prompt,
        encoding="utf-8",
    )

    (
        pred_dir
        / "target_user_prompt.txt"
    ).write_text(
        target_user_prompt,
        encoding="utf-8",
    )

    (
        pred_dir
        / "initial_response_source.json"
    ).write_text(
        json.dumps(
            {
                "initial_response_source": source_label,
                "a0_called": a0_called,
                "a0_call_count": (
                    1 if a0_called else 0
                ),
                "bootstrap_shared_initial_response": (
                    source_label
                    == "shared_initial_response_bootstrap"
                ),
                "student_only_evaluation": (
                    student_only_evaluation
                ),
                "student_forward_verification_only": (
                    student_forward_verification_only
                ),
                "selection_metric": (
                    selection_metric
                ),
                "student_response_reused": (
                    student_response_reused
                ),
                "selection_skill_hash": result[
                    "selection_skill_hash"
                ],
                "selection_request_hash": result[
                    "selection_request_hash"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        pred_dir
        / "conversation.json"
    ).write_text(
        json.dumps(
            evidence_to_conversation(evidence),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if (
        student_only_evaluation
        or student_forward_verification_only
    ):
        (
            pred_dir
            / "student_response.json"
        ).write_text(
            json.dumps(
                {
                    "selection_eval_protocol": (
                        selection_eval_protocol(
                            request_metric
                        )
                    ),
                    "selection_metric": request_metric,
                    "response": evidence.y,
                    "pred": evidence.a,
                    "answer_extraction_fallback": (
                        answer_extraction_fallback
                    ),
                    "selection_skill_hash": (
                        result[
                            "selection_skill_hash"
                        ]
                    ),
                    "selection_request_hash": (
                        result[
                            "selection_request_hash"
                        ]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (
            pred_dir
            / "forward_evidence.json"
        ).write_text(
            json.dumps(
                evidence.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return result


def _run_forward_math_item_with_factory(
    *,
    item: Dict[str, Any],
    skill_content: str,
    out_dir: Path,
    runner_factory: RunnerFactory,
    rollout_system_template: str,
    shared_initial_response_map: Dict[str, Dict[str, Any]] | None,
    bootstrap_shared_initial_response: bool = True,
    student_only_evaluation: bool = False,
    student_forward_verification_only: bool = False,
    selection_metric: str | None = None,
    student_response_reuse_map: (
        Dict[str, Dict[str, Any]] | None
    ) = None,
) -> Dict[str, Any]:
    return _run_forward_math_item(
        item=item,
        skill_content=skill_content,
        out_dir=out_dir,
        runner=runner_factory(),
        rollout_system_template=(
            rollout_system_template
        ),
        shared_initial_response_map=(
            shared_initial_response_map
        ),
        bootstrap_shared_initial_response=(
            bootstrap_shared_initial_response
        ),
        student_only_evaluation=(
            student_only_evaluation
        ),
        student_forward_verification_only=(
            student_forward_verification_only
        ),
        selection_metric=selection_metric,
        student_response_reuse_map=(
            student_response_reuse_map
        ),
    )


def _ordered_completed_results(
    results_by_index: List[Dict[str, Any] | None],
) -> List[Dict[str, Any]]:
    return [
        result
        for result in results_by_index
        if result is not None
    ]


def _write_rollout_error(
    out_dir: Path,
    *,
    phase: str,
    item: Dict[str, Any],
    error: BaseException,
) -> None:
    sample_id = str(
        item.get(
            "id",
            "",
        )
    )

    payload = {
        "phase": phase,
        "sample_id": sample_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        ),
    }

    path = out_dir / "failed_rollout_error.json"
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "[S5 rollout error] "
        f"phase={phase} sample={sample_id} "
        f"error={type(error).__name__}: {error}",
        flush=True,
    )
    print(
        f"[S5 rollout error] detail={path}",
        flush=True,
    )


def _candidate_hard_upper_bound(
    *,
    hard_sum: float,
    completed: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    remaining = max(
        0,
        total - completed,
    )

    return (
        float(hard_sum)
        + float(remaining)
    ) / float(total)


def _adaptive_result_hard(
    result: Dict[str, Any],
) -> float | None:
    try:
        hard = float(
            result.get(
                "hard",
                0.0,
            )
        )

    except (TypeError, ValueError):
        return None

    if (
        not math.isfinite(hard)
        or hard < 0.0
        or hard > 1.0
    ):
        return None

    return hard


def _write_adaptive_selection_state(
    out_dir: Path,
    state: Dict[str, Any],
) -> None:
    (
        out_dir
        / "adaptive_selection.json"
    ).write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _normalise_adaptive_worker_stages(
    raw_stages: Any,
    *,
    rollout_workers: int,
    total_items: int,
) -> List[int]:
    if (
        not isinstance(
            raw_stages,
            (list, tuple),
        )
        or not raw_stages
    ):
        raise ValueError(
            "adaptive selection worker_stages must be "
            "a non-empty list of positive integers."
        )

    try:
        stages = [
            int(value)
            for value in raw_stages
        ]

    except (TypeError, ValueError) as error:
        raise ValueError(
            "adaptive selection worker_stages must contain "
            "only positive integers."
        ) from error

    if any(
        value <= 0
        for value in stages
    ):
        raise ValueError(
            "adaptive selection worker_stages must contain "
            "only positive integers."
        )

    if any(
        left >= right
        for left, right in zip(
            stages,
            stages[1:],
        )
    ):
        raise ValueError(
            "adaptive selection worker_stages must be "
            "strictly increasing."
        )

    effective: List[int] = []

    for value in stages:
        capped = min(
            value,
            rollout_workers,
            total_items,
        )

        if (
            capped > 0
            and capped not in effective
        ):
            effective.append(
                capped
            )

    return effective


def _adaptive_submission_order(
    items: List[Dict[str, Any]],
    priority_sample_ids,
) -> tuple[list[int], list[int], str]:
    original_order = list(range(len(items)))
    if priority_sample_ids is None:
        return original_order, [], ""
    if not isinstance(priority_sample_ids, (list, tuple)):
        return original_order, [], "invalid_priority_sample_ids"

    index_by_id: dict[str, int] = {}
    duplicate_item_ids: set[str] = set()
    for index, item in enumerate(items):
        sample_id = str(item.get("id", ""))
        if sample_id in index_by_id:
            duplicate_item_ids.add(sample_id)
            continue
        index_by_id[sample_id] = index

    priority_indices: list[int] = []
    seen_indices: set[int] = set()
    ignored_priority_ids = False
    for value in priority_sample_ids:
        sample_id = str(value)
        index = index_by_id.get(sample_id)
        if (
            index is None
            or sample_id in duplicate_item_ids
            or index in seen_indices
        ):
            ignored_priority_ids = True
            continue
        priority_indices.append(index)
        seen_indices.add(index)

    submission_order = [
        *priority_indices,
        *[
            index
            for index in original_order
            if index not in seen_indices
        ],
    ]
    fallback_reason = (
        "unknown_or_duplicate_priority_ids_ignored"
        if ignored_priority_ids
        else ""
    )
    return submission_order, priority_indices, fallback_reason


def _run_adaptive_candidate_selection(
    *,
    items: List[Dict[str, Any]],
    skill_content: str,
    out_dir: Path,
    runner: ForwardSkillRunner,
    runner_factory: RunnerFactory | None,
    rollout_system_template: str,
    shared_initial_response_map: Dict[str, Dict[str, Any]] | None,
    bootstrap_shared_initial_response: bool,
    student_only_evaluation: bool,
    student_response_reuse_map: Dict[str, Dict[str, Any]] | None,
    rollout_workers: int,
    adaptive_selection: Dict[str, Any],
    skill_count: int,
    judge_data_path: str,
    judge_dataset_key: str,
    metrics_model: str,
    judge_workers: int,
    phase: str,
    student_forward_verification_only: bool = False,
    strict_validation_metric: str = REAL_ACC,
) -> List[Dict[str, Any]]:
    total_items = len(items)

    try:
        history_hard = float(
            adaptive_selection[
                "history_hard"
            ]
        )
        epsilon = float(
            adaptive_selection.get(
                "epsilon",
                1e-6,
            )
        )

    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "adaptive selection requires finite "
            "history_hard and epsilon."
        ) from error

    if (
        not math.isfinite(history_hard)
        or history_hard < 0.0
        or history_hard > 1.0
    ):
        raise ValueError(
            "adaptive selection history_hard must be in [0, 1]."
        )

    if (
        not math.isfinite(epsilon)
        or epsilon < 0.0
    ):
        raise ValueError(
            "adaptive selection epsilon must be non-negative."
        )

    worker_stages = (
        _normalise_adaptive_worker_stages(
            adaptive_selection.get(
                "worker_stages",
                [8, 16, 32],
            ),
            rollout_workers=rollout_workers,
            total_items=total_items,
        )
        if total_items > 0
        else []
    )

    (
        submission_order,
        priority_indices,
        priority_fallback_reason,
    ) = _adaptive_submission_order(
        items,
        adaptive_selection.get("priority_sample_ids"),
    )
    priority_index_set = set(priority_indices)
    completed_priority_indices: set[int] = set()
    adaptive_selection.update(
        {
            "priority_applied": bool(priority_indices),
            "priority_prefix_count": len(priority_indices),
            "priority_completed": 0,
            "priority_fallback_reason": priority_fallback_reason,
            "submission_order_sample_ids": [
                str(items[index].get("id", ""))
                for index in submission_order
            ],
        }
    )

    if total_items == 0:
        adaptive_selection.update(
            {
                "enabled": True,
                "metric": strict_validation_metric,
                "selection_eval_protocol": (
                    selection_eval_protocol(
                        strict_validation_metric
                    )
                ),
                "worker_stages": list(
                    adaptive_selection.get(
                        "worker_stages",
                        [8, 16, 32],
                    )
                ),
                "effective_worker_stages": [],
                "history_hard": history_hard,
                "history_score": history_hard,
                "epsilon": epsilon,
                "total_items": 0,
                "submitted_items": 0,
                "completed_items": 0,
                "hard_sum": 0.0,
                "partial_hard": 0.0,
                "hard_upper_bound": 0.0,
                "score_upper_bound": 0.0,
                "complete": True,
                "early_rejected": False,
                "stop_reason": "",
                "fallback_reason": "",
                "window_history": [],
            }
        )

        _write_adaptive_selection_state(
            out_dir,
            adaptive_selection,
        )

        return []

    max_workers = worker_stages[-1]

    if (
        max_workers > 1
        and runner_factory is None
    ):
        raise ValueError(
            "Parallel adaptive forward_math rollout requires "
            "runner_factory. Do not share one ForwardSkillRunner "
            "across worker threads."
        )

    results_by_index: List[
        Dict[str, Any] | None
    ] = [
        None
        for _ in items
    ]

    current_stage = 0
    current_window = worker_stages[
        current_stage
    ]

    window_history = [
        {
            "completed_items": 0,
            "workers": current_window,
        }
    ]

    next_index = 0
    completed = 0
    hard_sum = 0.0
    hard_values_valid = True

    stop_submitting = False
    stop_reason = ""
    fallback_reason = ""

    executor = ThreadPoolExecutor(
        max_workers=max_workers
    )

    future_to_index = {}

    def submit_one(
        index: int,
    ):
        item = items[index]

        if runner_factory is not None:
            future = executor.submit(
                _run_forward_math_item_with_factory,
                item=item,
                skill_content=skill_content,
                out_dir=out_dir,
                runner_factory=runner_factory,
                rollout_system_template=(
                    rollout_system_template
                ),
                shared_initial_response_map=(
                    shared_initial_response_map
                ),
                bootstrap_shared_initial_response=(
                    bootstrap_shared_initial_response
                ),
                student_only_evaluation=(
                    student_only_evaluation
                ),
                student_forward_verification_only=(
                    student_forward_verification_only
                ),
                selection_metric=(
                    strict_validation_metric
                ),
                student_response_reuse_map=(
                    student_response_reuse_map
                ),
            )

        else:
            future = executor.submit(
                _run_forward_math_item,
                item=item,
                skill_content=skill_content,
                out_dir=out_dir,
                runner=runner,
                rollout_system_template=(
                    rollout_system_template
                ),
                shared_initial_response_map=(
                    shared_initial_response_map
                ),
                bootstrap_shared_initial_response=(
                    bootstrap_shared_initial_response
                ),
                student_only_evaluation=(
                    student_only_evaluation
                ),
                student_forward_verification_only=(
                    student_forward_verification_only
                ),
                selection_metric=(
                    strict_validation_metric
                ),
                student_response_reuse_map=(
                    student_response_reuse_map
                ),
            )

        future_to_index[
            future
        ] = index

    def fill_window() -> None:
        nonlocal next_index

        while (
            not stop_submitting
            and next_index < total_items
            and len(future_to_index)
            < current_window
        ):
            submit_one(submission_order[next_index])
            next_index += 1

    try:
        fill_window()

        while future_to_index:
            done, _pending = wait(
                tuple(
                    future_to_index
                ),
                return_when=FIRST_COMPLETED,
            )

            completed_futures = sorted(
                done,
                key=lambda future: (
                    future_to_index[
                        future
                    ]
                ),
            )

            completed_batch: List[
                tuple[int, Dict[str, Any]]
            ] = []

            for future in completed_futures:
                index = future_to_index.pop(
                    future
                )

                try:
                    result = future.result()

                except BaseException as error:
                    _write_rollout_error(
                        out_dir,
                        phase=phase,
                        item=items[index],
                        error=error,
                    )
                    raise

                completed_batch.append(
                    (index, result)
                )

            if (
                student_only_evaluation
                or student_forward_verification_only
            ):
                _attach_selection_metric(
                    results=[
                        result
                        for _, result in completed_batch
                    ],
                    strict_validation_metric=(
                        strict_validation_metric
                    ),
                    judge_data_path=judge_data_path,
                    judge_dataset_key=(
                        judge_dataset_key
                    ),
                    judge_workers=judge_workers,
                )

            for index, result in completed_batch:
                results_by_index[
                    index
                ] = result

                if index in priority_index_set:
                    completed_priority_indices.add(index)

                completed += 1

                hard = _adaptive_result_hard(
                    result
                )

                if hard is None:
                    hard_values_valid = False
                    fallback_reason = (
                        "invalid_hard_value"
                    )
                    stop_submitting = False
                    stop_reason = ""

                else:
                    hard_sum += hard

                _print_s5_progress(
                    phase=phase,
                    done=completed,
                    total=total_items,
                    sample_id=str(
                        items[index]["id"]
                    ),
                )

            if not hard_values_valid:
                current_stage = (
                    len(worker_stages)
                    - 1
                )
                current_window = (
                    worker_stages[
                        current_stage
                    ]
                )

            else:
                hard_upper_bound = (
                    _candidate_hard_upper_bound(
                        hard_sum=hard_sum,
                        completed=completed,
                        total=total_items,
                    )
                )

                if (
                    next_index < total_items
                    and hard_upper_bound
                    <= history_hard + epsilon
                ):
                    stop_submitting = True
                    stop_reason = (
                        "cannot_exceed_history"
                    )

                if (
                    not stop_submitting
                    and completed > 0
                ):
                    observed_hard = (
                        hard_sum
                        / completed
                    )
                    priority_evidence_complete = (
                        bool(priority_indices)
                        and len(completed_priority_indices)
                        >= len(priority_indices)
                    )
                    expansion_allowed = (
                        priority_evidence_complete
                        or (
                            not priority_indices
                            and observed_hard
                            > history_hard + epsilon
                        )
                    )

                    while (
                        current_stage + 1
                        < len(worker_stages)
                        and completed
                        >= worker_stages[
                            current_stage
                        ]
                        and expansion_allowed
                    ):
                        current_stage += 1
                        current_window = (
                            worker_stages[
                                current_stage
                            ]
                        )

                        window_history.append(
                            {
                                "completed_items": (
                                    completed
                                ),
                                "workers": (
                                    current_window
                                ),
                            }
                        )

            # Finish tasks that were already submitted. Only stop submitting
            # new work, so no sample is interrupted while writing artifacts.
            fill_window()

    except BaseException as error:
        for future in future_to_index:
            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        partial_results = (
            _ordered_completed_results(
                results_by_index
            )
        )

        hard_upper_bound = (
            _candidate_hard_upper_bound(
                hard_sum=hard_sum,
                completed=completed,
                total=total_items,
            )
            if hard_values_valid
            else None
        )

        adaptive_selection.update(
            {
                "enabled": True,
                "metric": strict_validation_metric,
                "selection_eval_protocol": (
                    selection_eval_protocol(
                        strict_validation_metric
                    )
                ),
                "worker_stages": list(
                    adaptive_selection.get(
                        "worker_stages",
                        [8, 16, 32],
                    )
                ),
                "effective_worker_stages": (
                    worker_stages
                ),
                "window_history": (
                    window_history
                ),
                "history_hard": history_hard,
                "history_score": history_hard,
                "epsilon": epsilon,
                "total_items": total_items,
                "submitted_items": next_index,
                "completed_items": completed,
                "priority_completed": len(
                    completed_priority_indices
                ),
                "hard_sum": hard_sum,
                "partial_hard": (
                    hard_sum / completed
                    if completed > 0
                    and hard_values_valid
                    else None
                ),
                "hard_upper_bound": (
                    hard_upper_bound
                ),
                "score_upper_bound": (
                    hard_upper_bound
                ),
                "complete": False,
                "early_rejected": False,
                "stop_reason": "error",
                "fallback_reason": (
                    fallback_reason
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            }
        )

        _write_adaptive_selection_state(
            out_dir,
            adaptive_selection,
        )

        _write_results_and_metrics(
            out_dir=out_dir,
            results=partial_results,
            status="partial",
            skill_count=skill_count,
            judge_data_path=judge_data_path,
            judge_dataset_key=judge_dataset_key,
            metrics_model=metrics_model,
            judge_workers=judge_workers,
        )

        raise

    else:
        executor.shutdown(
            wait=True,
        )

    results = _ordered_completed_results(
        results_by_index
    )

    complete = (
        len(results)
        == total_items
    )

    early_rejected = (
        not complete
        and stop_reason
        == "cannot_exceed_history"
    )

    if (
        not complete
        and not early_rejected
    ):
        raise RuntimeError(
            "Adaptive candidate selection ended without "
            "a complete result or a valid early-rejection proof."
        )

    hard_upper_bound = (
        _candidate_hard_upper_bound(
            hard_sum=hard_sum,
            completed=len(results),
            total=total_items,
        )
        if hard_values_valid
        else None
    )

    adaptive_selection.update(
        {
            "enabled": True,
            "metric": strict_validation_metric,
            "selection_eval_protocol": (
                selection_eval_protocol(
                    strict_validation_metric
                )
            ),
            "worker_stages": list(
                adaptive_selection.get(
                    "worker_stages",
                    [8, 16, 32],
                )
            ),
            "effective_worker_stages": (
                worker_stages
            ),
            "window_history": (
                window_history
            ),
            "history_hard": history_hard,
            "history_score": history_hard,
            "epsilon": epsilon,
            "total_items": total_items,
            "submitted_items": next_index,
            "completed_items": len(
                results
            ),
            "priority_completed": len(
                completed_priority_indices
            ),
            "hard_sum": hard_sum,
            "partial_hard": (
                hard_sum / len(results)
                if results
                and hard_values_valid
                else None
            ),
            "hard_upper_bound": (
                hard_upper_bound
            ),
            "score_upper_bound": (
                hard_upper_bound
            ),
            "complete": complete,
            "early_rejected": (
                early_rejected
            ),
            "stop_reason": (
                stop_reason
                if early_rejected
                else ""
            ),
            "fallback_reason": (
                fallback_reason
            ),
        }
    )

    _write_adaptive_selection_state(
        out_dir,
        adaptive_selection,
    )

    _write_results_and_metrics(
        out_dir=out_dir,
        results=results,
        status=(
            "complete"
            if complete
            else "early_rejected"
        ),
        skill_count=skill_count,
        judge_data_path=judge_data_path,
        judge_dataset_key=judge_dataset_key,
        metrics_model=metrics_model,
        judge_workers=judge_workers,
    )

    return results


def run_forward_math_batch(
    *,
    items: Iterable[Dict[str, Any]],
    skill_content: str,
    out_dir: str | Path,
    runner: ForwardSkillRunner,
    rollout_system_path: str | Path,
    shared_initial_response_map: Dict[str, Dict[str, Any]] | None = None,
    bootstrap_shared_initial_response: bool = True,
    runner_factory: RunnerFactory | None = None,
    rollout_workers: int = 1,
    judge_data_path: str = "",
    judge_dataset_key: str = "",
    metrics_model: str = "",
    judge_workers: int = 1,
    adaptive_selection: Dict[str, Any] | None = None,
    student_response_reuse_results: Iterable[Dict[str, Any]] | None = None,
    strict_validation_metric: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Run one S5 forward_math batch.

    In the bootstrap rollout, the caller may pass shared initial
    responses so the first S5 evidence is based on the original
    initial answer rather than an empty-skill A0 regeneration.

    In selection/test evaluation phases:

        A0(problem, current skill)
        -> one complete Forward verification

    when strict_validation_metric=forward_pass_rate, or:

        A0(problem, current skill)
        -> strict three-step answer judge

    when strict_validation_metric=real_acc.

    Forward correction is not called in either evaluation flow.
    Exact, protocol-tagged student responses may be reused fail-closed.

    In training/reflection phases, the original complete Forward and
    Correction trajectory remains unchanged.

    Student generation remains blind to the gold answer. The existing
    three-step judge reads gold data only after the Student has answered.

    This function intentionally does not reuse old results.jsonl.
    Re-running the same step recomputes the batch.

    rollout_workers=1 preserves the original serial behavior.
    rollout_workers>1 runs samples in parallel with independent
    ForwardSkillRunner instances created by runner_factory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    items = list(items)
    total_items = len(items)
    phase = _progress_phase(out_dir)
    skill_count = _count_skill_entries(
        skill_content
    )
    selection_evaluation_phase = (
        is_student_only_evaluation_phase(
            out_dir
        )
    )
    selection_metric = (
        normalize_strict_validation_metric(
            strict_validation_metric
            if strict_validation_metric is not None
            else REAL_ACC
        )
        if selection_evaluation_phase
        else None
    )
    student_only_evaluation = (
        selection_evaluation_phase
        and selection_metric == REAL_ACC
    )
    student_forward_verification_only = (
        selection_evaluation_phase
        and selection_metric == FORWARD_PASS_RATE
    )
    effective_judge_data_path = (
        ""
        if student_forward_verification_only
        else judge_data_path
    )
    student_response_reuse_map = (
        _build_student_response_reuse_map(
            student_response_reuse_results
        )
    )

    if (
        selection_evaluation_phase
        and student_response_reuse_map is not None
    ):
        raise ValueError(
            "student_response_reuse_results are reserved for the "
            "later complete Forward/Correction evidence rollout."
        )

    _print_s5_progress(
        phase=phase,
        done=0,
        total=total_items,
        sample_id="-",
    )

    rollout_system_template = load_rollout_system(
        rollout_system_path
    )

    try:
        worker_count = max(
            1,
            int(
                rollout_workers
                or 1
            ),
        )

    except (TypeError, ValueError) as error:
        raise ValueError(
            "forward_math rollout_workers must be a positive integer."
        ) from error

    if adaptive_selection is not None:
        if not isinstance(
            adaptive_selection,
            dict,
        ):
            raise TypeError(
                "adaptive_selection must be a dict or None."
            )

        return _run_adaptive_candidate_selection(
            items=items,
            skill_content=skill_content,
            out_dir=out_dir,
            runner=runner,
            runner_factory=runner_factory,
            rollout_system_template=(
                rollout_system_template
            ),
            shared_initial_response_map=(
                shared_initial_response_map
            ),
            bootstrap_shared_initial_response=(
                bootstrap_shared_initial_response
            ),
            student_only_evaluation=(
                student_only_evaluation
            ),
            student_forward_verification_only=(
                student_forward_verification_only
            ),
            strict_validation_metric=(
                selection_metric
            ),
            student_response_reuse_map=(
                student_response_reuse_map
            ),
            rollout_workers=worker_count,
            adaptive_selection=(
                adaptive_selection
            ),
            skill_count=skill_count,
            judge_data_path=effective_judge_data_path,
            judge_dataset_key=judge_dataset_key,
            metrics_model=metrics_model,
            judge_workers=judge_workers,
            phase=phase,
        )

    if (
        worker_count <= 1
        or total_items <= 1
    ):
        results: List[Dict[str, Any]] = []

        try:
            for item_index, item in enumerate(
                items,
                start=1,
            ):
                result = _run_forward_math_item(
                    item=item,
                    skill_content=skill_content,
                    out_dir=out_dir,
                    runner=runner,
                    rollout_system_template=(
                        rollout_system_template
                    ),
                    shared_initial_response_map=(
                        shared_initial_response_map
                    ),
                    bootstrap_shared_initial_response=(
                        bootstrap_shared_initial_response
                    ),
                    student_only_evaluation=(
                        student_only_evaluation
                    ),
                    student_forward_verification_only=(
                        student_forward_verification_only
                    ),
                    selection_metric=selection_metric,
                    student_response_reuse_map=(
                        student_response_reuse_map
                    ),
                )

                results.append(
                    result
                )

                _print_s5_progress(
                    phase=phase,
                    done=item_index,
                    total=total_items,
                    sample_id=str(
                        item["id"]
                    ),
                )

        except Exception:
            _write_results_and_metrics(
                out_dir=out_dir,
                results=results,
                status="partial",
                skill_count=skill_count,
                judge_data_path=effective_judge_data_path,
                judge_dataset_key=judge_dataset_key,
                metrics_model=metrics_model,
                judge_workers=judge_workers,
            )
            raise

        if selection_evaluation_phase:
            _attach_selection_metric(
                results=results,
                strict_validation_metric=selection_metric,
                judge_data_path=effective_judge_data_path,
                judge_dataset_key=(
                    judge_dataset_key
                ),
                judge_workers=judge_workers,
            )

        _write_results_and_metrics(
            out_dir=out_dir,
            results=results,
            status="complete",
            skill_count=skill_count,
            judge_data_path=effective_judge_data_path,
            judge_dataset_key=judge_dataset_key,
            metrics_model=metrics_model,
            judge_workers=judge_workers,
        )

        return results

    if runner_factory is None:
        raise ValueError(
            "Parallel forward_math rollout requires runner_factory. "
            "Do not share one ForwardSkillRunner across worker threads."
        )

    max_workers = min(
        worker_count,
        total_items,
    )

    results_by_index: List[Dict[str, Any] | None] = [
        None
        for _ in items
    ]

    completed = 0

    executor = ThreadPoolExecutor(
        max_workers=max_workers
    )

    future_to_index = {}

    try:
        future_to_index = {
            executor.submit(
                _run_forward_math_item_with_factory,
                item=item,
                skill_content=skill_content,
                out_dir=out_dir,
                runner_factory=runner_factory,
                rollout_system_template=(
                    rollout_system_template
                ),
                shared_initial_response_map=(
                    shared_initial_response_map
                ),
                bootstrap_shared_initial_response=(
                    bootstrap_shared_initial_response
                ),
                student_only_evaluation=(
                    student_only_evaluation
                ),
                student_forward_verification_only=(
                    student_forward_verification_only
                ),
                selection_metric=selection_metric,
                student_response_reuse_map=(
                    student_response_reuse_map
                ),
            ): index
            for index, item in enumerate(
                items
            )
        }

        for future in as_completed(
            future_to_index
        ):
            index = future_to_index[
                future
            ]

            try:
                result = future.result()
            except BaseException as error:
                _write_rollout_error(
                    out_dir,
                    phase=phase,
                    item=items[index],
                    error=error,
                )
                raise

            results_by_index[index] = result

            completed += 1

            _print_s5_progress(
                phase=phase,
                done=completed,
                total=total_items,
                sample_id=str(
                    items[index]["id"]
                ),
            )

    except BaseException:
        for future in future_to_index:
            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        _write_results_and_metrics(
            out_dir=out_dir,
            results=_ordered_completed_results(
                results_by_index
            ),
            status="partial",
            skill_count=skill_count,
            judge_data_path=effective_judge_data_path,
            judge_dataset_key=judge_dataset_key,
            metrics_model=metrics_model,
            judge_workers=judge_workers,
        )

        raise

    else:
        executor.shutdown(
            wait=True,
        )

    results = _ordered_completed_results(
        results_by_index
    )

    if len(results) != total_items:
        raise RuntimeError(
            "Parallel forward_math rollout did not produce one result "
            "per input item."
        )

    if selection_evaluation_phase:
        _attach_selection_metric(
            results=results,
            strict_validation_metric=selection_metric,
            judge_data_path=effective_judge_data_path,
            judge_dataset_key=(
                judge_dataset_key
            ),
            judge_workers=judge_workers,
        )

    _write_results_and_metrics(
        out_dir=out_dir,
        results=results,
        status="complete",
        skill_count=skill_count,
        judge_data_path=effective_judge_data_path,
        judge_dataset_key=judge_dataset_key,
        metrics_model=metrics_model,
        judge_workers=judge_workers,
    )

    return results
