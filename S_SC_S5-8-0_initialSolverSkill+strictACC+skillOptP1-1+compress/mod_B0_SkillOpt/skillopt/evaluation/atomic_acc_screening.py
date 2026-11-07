"""Three-tier atomic-skill materialization and strict validation screening.

The loose and compact layers are deterministic views over the same
step-base Skill.  The strict layer evaluates every Rank-selected atomic
skill in priority order.  Each strict candidate is rebuilt from the
unchanged step-base Skill plus the cumulative set of accepted atomic
skills so compound ``replace`` Edits do not lose their original target.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Callable

from skillopt.optimizer.atomic_skills import (
    build_atomic_skill_view,
    rebuild_patch_from_atomic_selection,
)
from skillopt.optimizer.skill import (
    apply_patch_with_report,
)
from skillopt.optimizer.structured_skills import (
    dump_skill_library,
    parse_skill_library_text,
    skill_behavior_hash,
    validate_skill_library,
    write_json_atomic,
)


SKILL_LIBRARY_SCHEMA_VERSION = (
    "s5_7_8_three_tier_skill_state_v1"
)
STRICT_SCREENING_PROTOCOL = (
    "ranked_atomic_strict_acc_v1"
)
STRICT_SCREENING_RESUME_SCHEMA = (
    "ranked_atomic_strict_acc_resume_v1"
)
STRICT_SCREENING_INPUT_SCHEMA = (
    "ranked_atomic_strict_acc_input_v1"
)
STRICT_SCREENING_TRIAL_SCHEMA = (
    "ranked_atomic_strict_acc_trial_v1"
)


def skill_hash(content: str) -> str:
    return skill_behavior_hash(
        parse_skill_library_text(content)
    )


def _write_json(
    path: str,
    payload: dict[str, Any],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _write_json_atomic(
    path: str,
    payload: dict[str, Any],
) -> None:
    temporary_path = path + ".tmp"
    _write_json(
        temporary_path,
        payload,
    )
    os.replace(
        temporary_path,
        path,
    )


def _write_jsonl_atomic(
    path: str,
    rows: list[Any],
) -> tuple[int, str]:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    content = "\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in rows
    )
    encoded = content.encode("utf-8")
    temporary_path = path + ".tmp"
    with open(
        temporary_path,
        "wb",
    ) as file:
        file.write(encoded)
    os.replace(
        temporary_path,
        path,
    )
    return (
        len(rows),
        hashlib.sha256(encoded).hexdigest(),
    )


def _load_json(
    path: str,
) -> dict[str, Any] | None:
    try:
        with open(
            path,
            encoding="utf-8",
        ) as file:
            payload = json.load(file)
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    return payload if isinstance(payload, dict) else None


def _load_jsonl_checkpoint(
    path: str,
    *,
    expected_count: int,
    expected_sha256: str,
) -> list[Any] | None:
    try:
        with open(
            path,
            "rb",
        ) as file:
            encoded = file.read()
    except OSError:
        return None

    if (
        hashlib.sha256(encoded).hexdigest()
        != expected_sha256
    ):
        return None

    rows: list[Any] = []
    try:
        for line in encoded.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    if len(rows) != expected_count:
        return None
    return rows


def _stable_hash(
    payload: Any,
) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_screening_input_payload(
    *,
    step: int,
    step_base_skill: str,
    update_mode: str,
    compact_selection_budget: int,
    strict_screening_budget: int,
    post_role_merged_patch: dict[str, Any],
    atomic_skill_view: dict[str, Any],
    ranked_patch: dict[str, Any],
    ranked_atomic_skill_ids: list[str],
) -> dict[str, Any]:
    try:
        step = int(step)
        compact_selection_budget = int(
            compact_selection_budget
        )
        strict_screening_budget = int(
            strict_screening_budget
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Strict screening input identity must use integer "
            "step and budgets."
        ) from error

    if (
        step <= 0
        or compact_selection_budget <= 0
        or strict_screening_budget <= 0
    ):
        raise ValueError(
            "Strict screening input step and budgets must be positive."
        )
    if str(update_mode).strip().lower() != "patch":
        raise ValueError(
            "Strict atomic screening input requires update_mode='patch'."
        )
    for name, value in (
        (
            "post_role_merged_patch",
            post_role_merged_patch,
        ),
        ("atomic_skill_view", atomic_skill_view),
        ("ranked_patch", ranked_patch),
    ):
        if not isinstance(value, dict):
            raise TypeError(
                f"{name} must be a dictionary."
            )
    if not isinstance(
        ranked_atomic_skill_ids,
        list,
    ):
        raise TypeError(
            "ranked_atomic_skill_ids must be a list."
        )
    if len(ranked_atomic_skill_ids) != len(set(ranked_atomic_skill_ids)):
        raise ValueError("ranked_atomic_skill_ids must be unique.")

    atomic_skills = atomic_skill_view.get(
        "atomic_skills"
    )
    if not isinstance(atomic_skills, list):
        raise ValueError(
            "atomic_skill_view must contain an 'atomic_skills' list."
        )
    known_ids = {
        str(item.get("atomic_skill_id", ""))
        for item in atomic_skills
        if isinstance(item, dict)
    }
    unknown_ids = [
        atomic_skill_id
        for atomic_skill_id in ranked_atomic_skill_ids
        if atomic_skill_id not in known_ids
    ]
    if unknown_ids:
        raise ValueError(
            "Unknown ranked atomic skill ids in Strict screening "
            f"input: {unknown_ids}"
        )

    rebuilt_atomic_skill_view = (
        build_atomic_skill_view(
            post_role_merged_patch
        )
    )
    if (
        _stable_hash(rebuilt_atomic_skill_view)
        != _stable_hash(atomic_skill_view)
    ):
        raise ValueError(
            "atomic_skill_view does not match the committed "
            "Post-Role merged patch."
        )

    payload = {
        "schema_version": STRICT_SCREENING_INPUT_SCHEMA,
        "skill_library_schema_version": (
            SKILL_LIBRARY_SCHEMA_VERSION
        ),
        "screening_protocol": (
            STRICT_SCREENING_PROTOCOL
        ),
        "step": step,
        "step_base_hash": skill_hash(step_base_skill),
        "update_mode": "patch",
        "compact_selection_budget": compact_selection_budget,
        "strict_screening_budget": strict_screening_budget,
        "post_role_merged_patch": post_role_merged_patch,
        "post_role_merged_patch_hash": (
            _stable_hash(post_role_merged_patch)
        ),
        "atomic_skill_view": atomic_skill_view,
        "atomic_skill_view_hash": (
            _stable_hash(atomic_skill_view)
        ),
        "ranked_patch": ranked_patch,
        "ranked_patch_hash": (
            _stable_hash(ranked_patch)
        ),
        "ranked_atomic_skill_ids": list(ranked_atomic_skill_ids),
        "strict_screened_atomic_skill_ids": list(
            ranked_atomic_skill_ids[:strict_screening_budget]
        ),
    }
    payload["snapshot_hash"] = _stable_hash(payload)
    return payload


def _validate_strict_screening_input_snapshot(
    payload: dict[str, Any],
) -> None:
    if payload.get("schema_version") != (
        STRICT_SCREENING_INPUT_SCHEMA
    ):
        raise RuntimeError(
            "Strict screening input snapshot schema mismatch."
        )
    if payload.get(
        "skill_library_schema_version"
    ) != SKILL_LIBRARY_SCHEMA_VERSION:
        raise RuntimeError(
            "Strict screening input snapshot Skill-library schema "
            "mismatch."
        )
    if payload.get("screening_protocol") != (
        STRICT_SCREENING_PROTOCOL
    ):
        raise RuntimeError(
            "Strict screening input snapshot protocol mismatch."
        )

    recorded_snapshot_hash = payload.get(
        "snapshot_hash"
    )
    content = {
        key: value
        for key, value in payload.items()
        if key != "snapshot_hash"
    }
    if (
        not isinstance(recorded_snapshot_hash, str)
        or recorded_snapshot_hash
        != _stable_hash(content)
    ):
        raise RuntimeError(
            "Strict screening input snapshot hash mismatch."
        )

    for value_key, hash_key in (
        (
            "post_role_merged_patch",
            "post_role_merged_patch_hash",
        ),
        (
            "atomic_skill_view",
            "atomic_skill_view_hash",
        ),
        ("ranked_patch", "ranked_patch_hash"),
    ):
        value = payload.get(value_key)
        if (
            not isinstance(value, dict)
            or payload.get(hash_key)
            != _stable_hash(value)
        ):
            raise RuntimeError(
                "Strict screening input snapshot embedded hash "
                f"mismatch for {value_key}."
            )

    ranked_atomic_skill_ids = payload.get(
        "ranked_atomic_skill_ids"
    )
    atomic_skill_view = payload[
        "atomic_skill_view"
    ]
    try:
        step = int(payload.get("step"))
        compact_selection_budget = int(
            payload.get(
                "compact_selection_budget"
            )
        )
        strict_screening_budget = int(
            payload.get(
                "strict_screening_budget"
            )
        )
        atomic_skills = atomic_skill_view[
            "atomic_skills"
        ]
        if (
            step <= 0
            or compact_selection_budget <= 0
            or strict_screening_budget <= 0
            or not isinstance(atomic_skills, list)
            or not isinstance(
                ranked_atomic_skill_ids,
                list,
            )
            or len(ranked_atomic_skill_ids)
            != len(set(ranked_atomic_skill_ids))
        ):
            raise ValueError
        known_ids = {
            str(item["atomic_skill_id"])
            for item in atomic_skills
            if isinstance(item, dict)
        }
        if any(
            atomic_skill_id not in known_ids
            for atomic_skill_id in (
                ranked_atomic_skill_ids
            )
        ):
            raise ValueError
        rebuilt_atomic_skill_view = (
            build_atomic_skill_view(
                payload[
                    "post_role_merged_patch"
                ]
            )
        )
        if (
            _stable_hash(rebuilt_atomic_skill_view)
            != _stable_hash(atomic_skill_view)
        ):
            raise ValueError
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "Strict screening input snapshot content is invalid."
        ) from error

    expected_screened_ids = (
        ranked_atomic_skill_ids[
            :strict_screening_budget
        ]
    )
    if payload.get(
        "strict_screened_atomic_skill_ids"
    ) != expected_screened_ids:
        raise RuntimeError(
            "Strict screening input snapshot ranked prefix mismatch."
        )


def load_strict_screening_input_snapshot(
    path: str,
    *,
    step: int,
    step_base_skill: str,
    update_mode: str,
    compact_selection_budget: int,
    strict_screening_budget: int,
) -> dict[str, Any] | None:
    """Load one committed Rank/Post-Role input or fail closed."""
    if not os.path.exists(path):
        return None
    payload = _load_json(path)
    if payload is None:
        raise RuntimeError(
            "Strict screening input snapshot is unreadable or not "
            "a JSON object."
        )
    _validate_strict_screening_input_snapshot(payload)
    expected = {
        "step": int(step),
        "step_base_hash": skill_hash(step_base_skill),
        "update_mode": str(update_mode).strip().lower(),
        "compact_selection_budget": int(compact_selection_budget),
        "strict_screening_budget": int(strict_screening_budget),
    }
    mismatches = [
        key for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Strict screening input snapshot identity mismatch: "
            + ", ".join(mismatches)
            + ". Start a new output directory; the committed "
            "snapshot will not be overwritten."
        )
    return payload


def commit_strict_screening_input_snapshot(
    path: str,
    *,
    step: int,
    step_base_skill: str,
    update_mode: str,
    compact_selection_budget: int,
    strict_screening_budget: int,
    post_role_merged_patch: dict[str, Any],
    atomic_skill_view: dict[str, Any],
    ranked_patch: dict[str, Any],
    ranked_atomic_skill_ids: list[str],
) -> dict[str, Any]:
    """Atomically commit the exact Strict Rank/Post-Role input once."""
    payload = _strict_screening_input_payload(
        step=step,
        step_base_skill=step_base_skill,
        update_mode=update_mode,
        compact_selection_budget=compact_selection_budget,
        strict_screening_budget=strict_screening_budget,
        post_role_merged_patch=post_role_merged_patch,
        atomic_skill_view=atomic_skill_view,
        ranked_patch=ranked_patch,
        ranked_atomic_skill_ids=ranked_atomic_skill_ids,
    )
    if os.path.exists(path):
        existing = _load_json(path)
        if existing is None:
            raise RuntimeError(
                "Existing Strict screening input snapshot is "
                "unreadable and will not be overwritten."
            )
        _validate_strict_screening_input_snapshot(existing)
        if existing != payload:
            raise RuntimeError(
                "Existing Strict screening input snapshot differs "
                "from the current Rank/Post-Role input and will not "
                "be overwritten."
            )
        return existing
    _write_json_atomic(path, payload)
    return payload


def _write_skill(
    path: str,
    content: str,
) -> None:
    library = parse_skill_library_text(content)
    write_json_atomic(path, library)


def _normalise_evidence(
    evidence: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise TypeError(
            f"{name} must be a dictionary."
        )

    if "score" not in evidence:
        raise ValueError(
            f"{name} must contain 'score'."
        )

    try:
        score = float(evidence["score"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name}['score'] must be a finite number."
        ) from error

    if not math.isfinite(score):
        raise ValueError(
            f"{name}['score'] must be a finite number."
        )

    normalised = dict(evidence)
    normalised["score"] = score
    normalised.setdefault(
        "results",
        None,
    )
    normalised.setdefault(
        "prediction_dir",
        None,
    )
    return normalised


def _safe_trial_name(
    rank_index: int,
    atomic_skill_id: str,
) -> str:
    safe_id = "".join(
        character
        if character.isalnum()
        or character in {
            "-",
            "_",
        }
        else "_"
        for character in atomic_skill_id
    )
    return (
        f"trial_{rank_index:02d}_"
        f"{safe_id}"
    )


def _evaluation_summary(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in evidence.items()
        if key not in {
            "results",
        }
    }


def _resume_manifest_payload(
    *,
    step_base_skill: str,
    ranked_atomic_skill_ids: list[str],
    baseline_evidence: dict[str, Any],
    epsilon: float,
    screening_metric: str,
    resume_identity: dict[str, Any],
) -> dict[str, Any]:
    baseline_identity = {
        key: value
        for key, value in baseline_evidence.items()
        if key != "prediction_dir"
    }
    return {
        "schema_version": (
            STRICT_SCREENING_RESUME_SCHEMA
        ),
        "screening_protocol": (
            STRICT_SCREENING_PROTOCOL
        ),
        "step_base_hash": skill_hash(
            step_base_skill
        ),
        "ordered_atomic_skill_ids": list(
            ranked_atomic_skill_ids
        ),
        "epsilon": float(epsilon),
        "screening_metric": screening_metric,
        "baseline_evidence_hash": (
            _stable_hash(baseline_identity)
        ),
        "evaluation_identity": dict(
            resume_identity
        ),
        "evaluation_identity_hash": (
            _stable_hash(resume_identity)
        ),
    }


def _restore_strict_trial_prefix(
    *,
    step_base_skill: str,
    atomic_skill_view: dict[str, Any],
    ranked_atomic_skill_ids: list[str],
    baseline_evidence: dict[str, Any],
    epsilon: float,
    screening_metric: str,
    artifact_root: str,
    manifest_hash: str,
) -> dict[str, Any]:
    working_skill = step_base_skill
    working_evidence = dict(
        baseline_evidence
    )
    accepted_atomic_skill_ids: list[str] = []
    rejected_atomic_skill_ids: list[str] = []
    trials: list[dict[str, Any]] = []

    def fail_closed(
        rank_index: int,
        reason: str,
    ) -> None:
        raise RuntimeError(
            "Committed Strict trial cannot be resumed at "
            f"rank {rank_index}: {reason}. Existing artifacts "
            "will not be overwritten."
        )

    for rank_index, atomic_skill_id in enumerate(
        ranked_atomic_skill_ids,
        start=1,
    ):
        trial_dir = os.path.join(
            artifact_root,
            _safe_trial_name(
                rank_index,
                atomic_skill_id,
            ),
        )
        trial_record_path = os.path.join(
            trial_dir,
            "trial_result.json",
        )
        if not os.path.exists(
            trial_record_path
        ):
            for later_rank, later_id in enumerate(
                ranked_atomic_skill_ids[
                    rank_index:
                ],
                start=rank_index + 1,
            ):
                later_record_path = os.path.join(
                    artifact_root,
                    _safe_trial_name(
                        later_rank,
                        later_id,
                    ),
                    "trial_result.json",
                )
                if os.path.exists(
                    later_record_path
                ):
                    fail_closed(
                        rank_index,
                        "the committed trial sequence is not "
                        "a contiguous prefix",
                    )
            break
        trial_record = _load_json(
            trial_record_path
        )
        if trial_record is None:
            fail_closed(
                rank_index,
                "trial_result.json is unreadable or not a "
                "JSON object",
            )

        cumulative_ids = [
            *accepted_atomic_skill_ids,
            atomic_skill_id,
        ]
        candidate_patch = (
            rebuild_patch_from_atomic_selection(
                atomic_skill_view,
                cumulative_ids,
                reasoning=(
                    "Strict validation screening of "
                    "Rank-selected atomic skills."
                ),
                ranking_details={
                    "ranking_unit": (
                        "atomic_json_skill"
                    ),
                    "selected_atomic_skill_ids": (
                        cumulative_ids
                    ),
                },
            )
        )
        candidate_skill, _ = (
            apply_patch_with_report(
                step_base_skill,
                candidate_patch,
            )
        )
        candidate_hash = skill_hash(
            candidate_skill
        )
        score_before = float(
            working_evidence["score"]
        )

        try:
            record_rank = int(
                trial_record.get("rank")
            )
            record_score_before = float(
                trial_record.get(
                    "score_before",
                    trial_record.get("acc_before"),
                )
            )
            record_candidate_score = float(
                trial_record.get(
                    "candidate_score",
                    trial_record.get("candidate_acc"),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            fail_closed(
                rank_index,
                "numeric fields are invalid",
            )

        evaluation = trial_record.get(
            "evaluation"
        )
        if not isinstance(evaluation, dict):
            fail_closed(
                rank_index,
                "evaluation is not a JSON object",
            )
        try:
            candidate_evidence = (
                _normalise_evidence(
                    evaluation,
                    name=(
                        "restored_candidate_evidence"
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            fail_closed(
                rank_index,
                "evaluation evidence is invalid",
            )

        complete = bool(
            candidate_evidence.get(
                "complete",
                False,
            )
        )
        early_rejected = bool(
            candidate_evidence.get(
                "early_rejected",
                False,
            )
        )
        if not complete and not early_rejected:
            fail_closed(
                rank_index,
                "evaluation is neither complete nor a valid "
                "early rejection",
            )

        expected_action = (
            "accept"
            if (
                complete
                and candidate_evidence["score"]
                > score_before + epsilon
            )
            else "reject"
        )
        expected_score_kind = str(
            candidate_evidence.get(
                "score_kind",
                "complete"
                if complete
                else "upper_bound",
            )
        )
        record_score_kind = str(
            trial_record.get("candidate_score_kind")
            if "candidate_score_kind" in trial_record
            else trial_record.get(
                "candidate_acc_kind",
                "",
            )
        )
        record_screening_metric = str(
            trial_record.get(
                "screening_metric",
                "real_acc",
            )
        ).strip().lower()
        if (
            trial_record.get("schema_version")
            != STRICT_SCREENING_TRIAL_SCHEMA
            or trial_record.get("trial_status")
            != "complete"
            or trial_record.get(
                "resume_manifest_hash"
            )
            != manifest_hash
            or record_rank != rank_index
            or trial_record.get(
                "atomic_skill_id"
            )
            != atomic_skill_id
            or trial_record.get(
                "working_hash_before"
            )
            != skill_hash(working_skill)
            or trial_record.get(
                "cumulative_atomic_skill_ids"
            )
            != cumulative_ids
            or trial_record.get(
                "candidate_hash"
            )
            != candidate_hash
            or record_score_before != score_before
            or record_candidate_score
            != candidate_evidence["score"]
            or record_score_kind != expected_score_kind
            or record_screening_metric != screening_metric
            or trial_record.get("action")
            != expected_action
        ):
            fail_closed(
                rank_index,
                "schema, manifest hash, candidate lineage, score, "
                "or action mismatch",
            )

        results_checkpoint = trial_record.get(
            "evidence_results_checkpoint"
        )
        restored_results = None
        if isinstance(
            results_checkpoint,
            dict,
        ):
            checkpoint_name = str(
                results_checkpoint.get(
                    "path",
                    "",
                )
            )
            if checkpoint_name != (
                "trial_evidence_results.jsonl"
            ):
                fail_closed(
                    rank_index,
                    "evidence checkpoint path is invalid",
                )
            try:
                expected_count = int(
                    results_checkpoint.get(
                        "count"
                    )
                )
                expected_sha256 = str(
                    results_checkpoint.get(
                        "sha256"
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                fail_closed(
                    rank_index,
                    "evidence checkpoint metadata is invalid",
                )
            restored_results = (
                _load_jsonl_checkpoint(
                    os.path.join(
                        trial_dir,
                        checkpoint_name,
                    ),
                    expected_count=(
                        expected_count
                    ),
                    expected_sha256=(
                        expected_sha256
                    ),
                )
            )
            if restored_results is None:
                fail_closed(
                    rank_index,
                    "evidence checkpoint hash or row count "
                    "mismatch",
                )

        if complete and restored_results is None:
            fail_closed(
                rank_index,
                "complete evaluation has no committed results "
                "checkpoint",
            )

        if expected_action == "accept":
            candidate_evidence["results"] = (
                restored_results
            )
            accepted_atomic_skill_ids.append(
                atomic_skill_id
            )
            working_skill = candidate_skill
            working_evidence = (
                candidate_evidence
            )
        else:
            rejected_atomic_skill_ids.append(
                atomic_skill_id
            )
        trials.append(trial_record)

    return {
        "working_skill": working_skill,
        "working_evidence": working_evidence,
        "accepted_atomic_skill_ids": (
            accepted_atomic_skill_ids
        ),
        "rejected_atomic_skill_ids": (
            rejected_atomic_skill_ids
        ),
        "trials": trials,
    }


def materialize_patch_skill_layers(
    *,
    step_base_skill: str,
    loose_patch: dict,
    compact_patch: dict,
) -> dict[str, Any]:
    """Apply loose and compact patches independently to one base Skill."""
    loose_skill, loose_apply_report = (
        apply_patch_with_report(
            step_base_skill,
            loose_patch,
        )
    )
    compact_skill, compact_apply_report = (
        apply_patch_with_report(
            step_base_skill,
            compact_patch,
        )
    )

    return {
        "loose_skill": loose_skill,
        "loose_hash": skill_hash(
            loose_skill
        ),
        "loose_apply_report": (
            loose_apply_report
        ),
        "compact_skill": compact_skill,
        "compact_hash": skill_hash(
            compact_skill
        ),
        "compact_apply_report": (
            compact_apply_report
        ),
    }


def persist_three_tier_skill_state(
    *,
    step_dir: str,
    step: int,
    active_mode: str,
    base_skill: str,
    loose_skill: str | None,
    compact_skill: str | None,
    strict_skill: str | None,
    loose_status: str,
    compact_status: str,
    strict_status: str,
    compact_selection_budget: int | None = None,
    strict_screening_budget: int | None = None,
    strict_tested_atomic_skill_count: int | None = None,
) -> dict[str, Any]:
    """Persist one normalized three-tier state without duplicating skills."""
    active_mode = str(active_mode).strip().lower()
    if active_mode not in {"loose", "compact", "strict"}:
        raise ValueError(
            "active_mode must be 'loose', 'compact', or 'strict'."
        )

    for field_name, value in (
        ("compact_selection_budget", compact_selection_budget),
        ("strict_screening_budget", strict_screening_budget),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a positive integer or None."
            )

    if strict_tested_atomic_skill_count is not None and (
        isinstance(strict_tested_atomic_skill_count, bool)
        or not isinstance(strict_tested_atomic_skill_count, int)
        or strict_tested_atomic_skill_count < 0
    ):
        raise ValueError(
            "strict_tested_atomic_skill_count must be a non-negative "
            "integer or None."
        )

    layer_inputs = {
        "loose": (loose_status, loose_skill),
        "compact": (compact_status, compact_skill),
        "strict": (strict_status, strict_skill),
    }
    skill_by_id: dict[str, dict[str, Any]] = {}
    layers: dict[str, dict[str, Any]] = {}

    for layer_name, (status, content) in layer_inputs.items():
        ordered_skill_ids: list[str] = []
        if content is not None:
            library = parse_skill_library_text(content)
            for skill in library["skills"]:
                skill_id = str(skill["skill_id"])
                ordered_skill_ids.append(skill_id)
                skill_by_id.setdefault(skill_id, dict(skill))
        layer_record: dict[str, Any] = {
            "status": str(status),
            "ordered_skill_ids": ordered_skill_ids,
        }
        if layer_name == "compact":
            layer_record["selection_budget"] = compact_selection_budget
        elif layer_name == "strict":
            layer_record["screening_budget"] = strict_screening_budget
            layer_record["tested_atomic_skill_count"] = (
                strict_tested_atomic_skill_count
            )
        layers[layer_name] = layer_record

    state = {
        "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
        "step": int(step),
        "base_behavior_hash": skill_hash(base_skill),
        "active_mode": active_mode,
        "skills": list(skill_by_id.values()),
        "layers": layers,
    }
    write_json_atomic(
        os.path.join(step_dir, "skill_state.json"),
        state,
    )
    return state


def run_ranked_atomic_strict_screening(
    *,
    step_base_skill: str,
    atomic_skill_view: dict[str, Any],
    ranked_atomic_skill_ids: list[str],
    baseline_evidence: dict[str, Any],
    epsilon: float,
    screening_metric: str = "real_acc",
    artifact_root: str,
    evaluate_candidate: Callable[
        [
            str,
            str,
            float,
            list[dict] | None,
        ],
        dict[str, Any],
    ],
    resume_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly screen every ranked atomic skill against the live baseline.

    Passing ``resume_identity`` enables safe reuse of the continuous prefix
    of fully committed trials.  Omitting it preserves the original fresh-run
    behaviour.
    """
    if not isinstance(
        atomic_skill_view,
        dict,
    ):
        raise TypeError(
            "atomic_skill_view must be a dictionary."
        )

    atomic_skills = atomic_skill_view.get(
        "atomic_skills"
    )
    if not isinstance(
        atomic_skills,
        list,
    ):
        raise ValueError(
            "atomic_skill_view must contain an "
            "'atomic_skills' list."
        )

    screening_metric = str(
        screening_metric
    ).strip().lower()
    if not screening_metric:
        raise ValueError(
            "screening_metric must be a non-empty string."
        )

    if not isinstance(
        ranked_atomic_skill_ids,
        list,
    ):
        raise TypeError(
            "ranked_atomic_skill_ids must be a list."
        )

    if len(
        ranked_atomic_skill_ids
    ) != len(
        set(ranked_atomic_skill_ids)
    ):
        raise ValueError(
            "ranked_atomic_skill_ids must be unique."
        )

    known_ids = {
        str(
            item.get(
                "atomic_skill_id",
                "",
            )
        )
        for item in atomic_skills
        if isinstance(item, dict)
    }
    unknown_ids = [
        atomic_skill_id
        for atomic_skill_id in (
            ranked_atomic_skill_ids
        )
        if atomic_skill_id not in known_ids
    ]
    if unknown_ids:
        raise ValueError(
            "Unknown ranked atomic skill ids: "
            f"{unknown_ids}"
        )

    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "epsilon must be a finite, non-negative number."
        ) from error

    if (
        not math.isfinite(epsilon)
        or epsilon < 0.0
    ):
        raise ValueError(
            "epsilon must be a finite, non-negative number."
        )

    if (
        resume_identity is not None
        and not isinstance(
            resume_identity,
            dict,
        )
    ):
        raise TypeError(
            "resume_identity must be a dictionary or None."
        )

    working_evidence = _normalise_evidence(
        baseline_evidence,
        name="baseline_evidence",
    )
    working_skill = step_base_skill
    accepted_atomic_skill_ids: list[str] = []
    rejected_atomic_skill_ids: list[str] = []
    trials: list[dict[str, Any]] = []

    os.makedirs(
        artifact_root,
        exist_ok=True,
    )

    resume_enabled = resume_identity is not None
    resume_manifest_path = os.path.join(
        artifact_root,
        "resume_manifest.json",
    )
    resume_manifest = None
    resume_manifest_hash = ""
    resumed_trial_count = 0

    if resume_enabled:
        resume_manifest = (
            _resume_manifest_payload(
                step_base_skill=(
                    step_base_skill
                ),
                ranked_atomic_skill_ids=(
                    ranked_atomic_skill_ids
                ),
                baseline_evidence=(
                    working_evidence
                ),
                epsilon=epsilon,
                screening_metric=(
                    screening_metric
                ),
                resume_identity=(
                    resume_identity
                ),
            )
        )
        resume_manifest_hash = _stable_hash(
            resume_manifest
        )
        if os.path.exists(
            resume_manifest_path
        ):
            existing_manifest = _load_json(
                resume_manifest_path
            )
            if existing_manifest is None:
                raise RuntimeError(
                    "Existing Strict resume manifest is unreadable "
                    "and will not be overwritten."
                )
            if existing_manifest != resume_manifest:
                raise RuntimeError(
                    "Existing Strict resume manifest does not match "
                    "the current screening input/evaluation identity. "
                    "Start a new output directory; existing artifacts "
                    "will not be overwritten."
                )
            restored = (
                _restore_strict_trial_prefix(
                    step_base_skill=(
                        step_base_skill
                    ),
                    atomic_skill_view=(
                        atomic_skill_view
                    ),
                    ranked_atomic_skill_ids=(
                        ranked_atomic_skill_ids
                    ),
                    baseline_evidence=(
                        working_evidence
                    ),
                    epsilon=epsilon,
                    screening_metric=(
                        screening_metric
                    ),
                    artifact_root=(
                        artifact_root
                    ),
                    manifest_hash=(
                        resume_manifest_hash
                    ),
                )
            )
            working_skill = restored[
                "working_skill"
            ]
            working_evidence = restored[
                "working_evidence"
            ]
            accepted_atomic_skill_ids = (
                restored[
                    "accepted_atomic_skill_ids"
                ]
            )
            rejected_atomic_skill_ids = (
                restored[
                    "rejected_atomic_skill_ids"
                ]
            )
            trials = restored["trials"]
            resumed_trial_count = len(trials)
        else:
            _write_json_atomic(
                resume_manifest_path,
                resume_manifest,
            )

    for rank_index, atomic_skill_id in enumerate(
        ranked_atomic_skill_ids[
            resumed_trial_count:
        ],
        start=resumed_trial_count + 1,
    ):
        cumulative_ids = [
            *accepted_atomic_skill_ids,
            atomic_skill_id,
        ]
        candidate_patch = (
            rebuild_patch_from_atomic_selection(
                atomic_skill_view,
                cumulative_ids,
                reasoning=(
                    "Strict validation screening of "
                    "Rank-selected atomic skills."
                ),
                ranking_details={
                    "ranking_unit": (
                        "atomic_json_skill"
                    ),
                    "selected_atomic_skill_ids": (
                        cumulative_ids
                    ),
                },
            )
        )
        (
            candidate_skill,
            candidate_apply_report,
        ) = apply_patch_with_report(
            step_base_skill,
            candidate_patch,
        )
        candidate_hash = skill_hash(
            candidate_skill
        )
        score_before = float(
            working_evidence["score"]
        )
        trial_dir = os.path.join(
            artifact_root,
            _safe_trial_name(
                rank_index,
                atomic_skill_id,
            ),
        )
        candidate_path = os.path.join(
            trial_dir,
            "candidate_skill.json",
        )
        _write_skill(
            candidate_path,
            candidate_skill,
        )

        if candidate_skill == working_skill:
            candidate_evidence = {
                "score": score_before,
                "complete": False,
                "early_rejected": True,
                "score_kind": "no_effect",
                "evaluated_samples": 0,
                "stop_reason": "apply_no_effect",
            }
        elif score_before >= 1.0 - epsilon:
            candidate_evidence = {
                "score": 1.0,
                "complete": False,
                "early_rejected": True,
                "score_kind": "mathematical_upper_bound",
                "evaluated_samples": 0,
                "stop_reason": "history_acc_at_ceiling",
            }
        else:
            raw_candidate_evidence = (
                evaluate_candidate(
                    candidate_skill,
                    trial_dir,
                    score_before,
                    working_evidence.get(
                        "results"
                    ),
                )
            )
            candidate_evidence = (
                _normalise_evidence(
                    raw_candidate_evidence,
                    name=(
                        "candidate_evidence"
                    ),
                )
            )
        complete = bool(
            candidate_evidence.get(
                "complete",
                False,
            )
        )
        early_rejected = bool(
            candidate_evidence.get(
                "early_rejected",
                False,
            )
        )
        if not complete and not early_rejected:
            raise RuntimeError(
                "Atomic strict-validation evaluation must return either a "
                "complete result or a valid early-rejection proof."
            )

        candidate_score = float(
            candidate_evidence["score"]
        )
        candidate_score_kind = str(
            candidate_evidence.get(
                "score_kind",
                "complete"
                if complete
                else "upper_bound",
            )
        )
        accepted = (
            complete
            and candidate_score > score_before + epsilon
        )
        action = "accept" if accepted else "reject"

        results_checkpoint = None
        evidence_results = candidate_evidence.get(
            "results"
        )
        if isinstance(evidence_results, list):
            results_path = os.path.join(
                trial_dir,
                "trial_evidence_results.jsonl",
            )
            results_count, results_sha256 = (
                _write_jsonl_atomic(
                    results_path,
                    evidence_results,
                )
            )
            results_checkpoint = {
                "path": (
                    "trial_evidence_results.jsonl"
                ),
                "count": results_count,
                "sha256": results_sha256,
            }

        trial_record = {
            "schema_version": (
                STRICT_SCREENING_TRIAL_SCHEMA
            ),
            "trial_status": "complete",
            "rank": rank_index,
            "atomic_skill_id": (
                atomic_skill_id
            ),
            "atomic_skill": next(
                dict(item)
                for item in atomic_skills
                if str(item.get("atomic_skill_id"))
                == atomic_skill_id
            ),
            "cumulative_atomic_skill_ids": (
                cumulative_ids
            ),
            "working_hash_before": skill_hash(
                working_skill
            ),
            "candidate_hash": candidate_hash,
            "candidate_skill_path": (
                candidate_path
            ),
            "artifact_dir": trial_dir,
            "screening_metric": screening_metric,
            "score_before": score_before,
            "candidate_score": candidate_score,
            "candidate_score_kind": (
                candidate_score_kind
            ),
            "acc_before": score_before,
            "candidate_acc": candidate_score,
            "delta": candidate_score - score_before,
            "candidate_acc_kind": candidate_score_kind,
            "complete": complete,
            "early_rejected": (
                early_rejected
            ),
            "action": action,
            "stop_reason": candidate_evidence.get(
                "stop_reason",
                "",
            ),
            "evaluated_samples": int(
                candidate_evidence.get(
                    "evaluated_samples",
                    0,
                )
                or 0
            ),
            "selection_eval_protocol": (
                candidate_evidence.get(
                    "selection_eval_protocol"
                )
            ),
            "apply_report": (
                candidate_apply_report
            ),
            "evaluation": (
                _evaluation_summary(
                    candidate_evidence
                )
            ),
            "evidence_results_checkpoint": (
                results_checkpoint
            ),
        }
        if resume_enabled:
            trial_record[
                "resume_manifest_hash"
            ] = resume_manifest_hash

        _write_json_atomic(
            os.path.join(
                trial_dir,
                "trial_result.json",
            ),
            trial_record,
        )

        if accepted:
            accepted_atomic_skill_ids.append(
                atomic_skill_id
            )
            working_skill = candidate_skill
            working_evidence = dict(
                candidate_evidence
            )
        else:
            rejected_atomic_skill_ids.append(
                atomic_skill_id
            )
        trials.append(
            trial_record
        )

    final_patch = rebuild_patch_from_atomic_selection(
        atomic_skill_view,
        accepted_atomic_skill_ids,
        reasoning=(
            "Strict validation accepted Rank-selected atomic skills."
        ),
        ranking_details={
            "ranking_unit": "atomic_json_skill",
            "selected_atomic_skill_ids": list(
                accepted_atomic_skill_ids
            ),
        },
    )
    rebuilt_final_skill, final_apply_report = (
        apply_patch_with_report(
            step_base_skill,
            final_patch,
        )
    )
    if rebuilt_final_skill != working_skill:
        raise RuntimeError(
            "Strict screening could not reproduce its final Skill "
            "from the step base and accepted atomic-skill ids."
        )

    report = {
        "schema_version": (
            SKILL_LIBRARY_SCHEMA_VERSION
        ),
        "protocol": (
            STRICT_SCREENING_PROTOCOL
        ),
        "screening_metric": screening_metric,
        "selection_eval_protocol": (
            working_evidence.get(
                "selection_eval_protocol"
            )
            or baseline_evidence.get(
                "selection_eval_protocol"
            )
        ),
        "step_base_hash": skill_hash(
            step_base_skill
        ),
        "ordered_atomic_skill_ids": list(
            ranked_atomic_skill_ids
        ),
        "accepted_atomic_skill_ids": list(
            accepted_atomic_skill_ids
        ),
        "rejected_atomic_skill_ids": list(
            rejected_atomic_skill_ids
        ),
        "baseline_acc": float(
            baseline_evidence["score"]
        ),
        "final_acc": float(
            working_evidence["score"]
        ),
        "baseline_score": float(
            baseline_evidence["score"]
        ),
        "final_score": float(
            working_evidence["score"]
        ),
        "final_hash": skill_hash(working_skill),
        "resume": {
            "enabled": resume_enabled,
            "resumed_trial_count": (
                resumed_trial_count
            ),
            "manifest": (
                "strict_trials/resume_manifest.json"
                if resume_enabled
                else None
            ),
            "manifest_hash": (
                resume_manifest_hash
                if resume_enabled
                else None
            ),
        },
        "trials": trials,
    }
    _write_json_atomic(
        os.path.join(
            os.path.dirname(artifact_root),
            "strict_atomic_screening.json",
        ),
        report,
    )

    return {
        "final_skill": working_skill,
        "final_patch": final_patch,
        "final_apply_report": final_apply_report,
        "final_evidence": (
            working_evidence
        ),
        "accepted_atomic_skill_ids": (
            accepted_atomic_skill_ids
        ),
        "rejected_atomic_skill_ids": (
            rejected_atomic_skill_ids
        ),
        "report": report,
    }


def persist_skill_library_manifest(
    *,
    out_root: str,
    active_mode: str,
    step_record: dict[str, Any],
) -> None:
    """Upsert one compact per-step entry in the root library manifest."""
    active_mode = str(
        active_mode
    ).strip().lower()
    if active_mode not in {
        "loose",
        "compact",
        "strict",
    }:
        raise ValueError(
            "active_mode must be 'loose', "
            "'compact', or 'strict'."
        )

    if not isinstance(
        step_record,
        dict,
    ):
        raise TypeError(
            "step_record must be a dictionary."
        )

    library_dir = os.path.join(
        out_root,
        "skill_libraries",
    )
    manifest_path = os.path.join(
        library_dir,
        "manifest.json",
    )
    os.makedirs(
        library_dir,
        exist_ok=True,
    )

    if os.path.exists(
        manifest_path
    ):
        with open(
            manifest_path,
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if (
            manifest.get(
                "schema_version"
            )
            != SKILL_LIBRARY_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "Cannot update a skill-library manifest "
                "with an incompatible schema version."
            )

        restored_mode = str(
            manifest.get(
                "active_mode",
                "",
            )
        ).strip().lower()
        if restored_mode != active_mode:
            raise RuntimeError(
                "Cannot resume a skill-library manifest "
                "with a different active mode."
            )

    else:
        manifest = {
            "schema_version": (
                SKILL_LIBRARY_SCHEMA_VERSION
            ),
            "active_mode": active_mode,
            "steps": [],
        }

    try:
        step = int(
            step_record["step"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "step_record must contain an integer 'step'."
        ) from error

    steps = [
        existing
        for existing in manifest.get(
            "steps",
            [],
        )
        if int(
            existing.get(
                "step",
                -1,
            )
        )
        != step
    ]
    steps.append(step_record)
    steps.sort(
        key=lambda value: int(
            value.get(
                "step",
                0,
            )
        )
    )
    manifest["steps"] = steps

    temporary_path = (
        manifest_path
        + ".tmp"
    )
    _write_json(
        temporary_path,
        manifest,
    )
    os.replace(
        temporary_path,
        manifest_path,
    )


__all__ = [
    "SKILL_LIBRARY_SCHEMA_VERSION",
    "STRICT_SCREENING_PROTOCOL",
    "STRICT_SCREENING_RESUME_SCHEMA",
    "STRICT_SCREENING_INPUT_SCHEMA",
    "STRICT_SCREENING_TRIAL_SCHEMA",
    "commit_strict_screening_input_snapshot",
    "load_strict_screening_input_snapshot",
    "materialize_patch_skill_layers",
    "persist_skill_library_manifest",
    "persist_three_tier_skill_state",
    "run_ranked_atomic_strict_screening",
]
