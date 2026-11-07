"""Run-local role approval reuse for S5 with the P1 optimizer."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from skillopt.evaluation.role_consistency import (
    enforce_structured_atomic_skill_role_consistency,
    structured_atomic_skill_error,
)
from skillopt.optimizer.structured_skills import (
    parse_patch_payload,
    skill_id_from_draft,
)
from skillopt.prompts import load_prompt


_ROLE_PROMPTS = (
    "role_consistency_role",
    "role_consistency_check",
    "role_consistency_repair",
    "role_consistency_postcheck",
)
_PASS = {"pass_without_repair", "pass_after_repair"}
_DROP = {"fail_after_repair", "unrepairable_no_commit"}


class RoleScreeningTechnicalError(RuntimeError):
    def __init__(self, message: str, *, audit: dict):
        super().__init__(message)
        self.audit = copy.deepcopy(audit)


class ApprovedRoleCache:
    """One instance per training run; never shared across experiments."""

    def __init__(self):
        self._approved = {}
        self._lock = threading.Lock()

    def _key(self, skill: dict, options: dict) -> str:
        payload = {
            "schema_version": "s5_role_approval_v1",
            "skill": skill,
            "options": options,
            "role_prompts": {
                name: load_prompt(name, env=options["env"])
                for name in _ROLE_PROMPTS
            },
        }
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def check(
        self,
        candidate_skill: dict,
        *,
        env="forward_math",
        max_completion_tokens=0,
        format_max_attempts=3,
    ) -> dict:
        options = {
            "env": env,
            "max_completion_tokens": max_completion_tokens,
            "format_max_attempts": format_max_attempts,
        }

        if structured_atomic_skill_error(candidate_skill) is not None:
            return enforce_structured_atomic_skill_role_consistency(
                copy.deepcopy(candidate_skill), **options
            )

        key = self._key(candidate_skill, options)
        with self._lock:
            approval = copy.deepcopy(self._approved.get(key))

        if approval is not None:
            return {
                "passed": True,
                "status": "pass_without_repair",
                "candidate_skill": copy.deepcopy(candidate_skill),
                "changed": False,
                "role_approval_cache": {
                    "hit": True,
                    "key": key,
                    "approved_result": approval,
                },
            }

        result = enforce_structured_atomic_skill_role_consistency(
            copy.deepcopy(candidate_skill), **options
        )
        if not isinstance(result, dict):
            raise TypeError("Role checker must return a dictionary.")

        output = result.get("candidate_skill")
        if (
            result.get("passed") is True
            and result.get("status") in _PASS
            and structured_atomic_skill_error(output) is None
        ):
            output_key = self._key(output, options)
            with self._lock:
                # Repair A -> B approves B, not the rejected original A.
                self._approved[output_key] = copy.deepcopy(result)

        result = copy.deepcopy(result)
        result["role_approval_cache"] = {
            "hit": False,
            "key": key,
        }
        return result


def filter_patch(
    cache: ApprovedRoleCache,
    patch: dict,
    *,
    env="forward_math",
    workers=1,
    max_completion_tokens=0,
    format_max_attempts=3,
    known_skill_ids=None,
) -> tuple[dict, dict]:
    if type(workers) is not int or workers < 1:
        raise ValueError("Role workers must be a positive integer.")

    parse_options = {
        "known_skill_ids": known_skill_ids,
        "allow_ranking_details": True,
        "allow_trusted_metadata": True,
    }

    try:
        canonical = parse_patch_payload(patch, **parse_options)
    except Exception as error:
        raise RoleScreeningTechnicalError(
            f"Invalid role-screening patch: {error}",
            audit={
                "status": "invalid_input_patch",
                "error": str(error),
            },
        ) from error

    items = [
        (edit_index, skill_index, skill)
        for edit_index, edit in enumerate(canonical["edits"])
        for skill_index, skill in enumerate(edit.get("skills", []))
    ]

    def check_one(item):
        edit_index, skill_index, skill = item
        try:
            result = cache.check(
                skill,
                env=env,
                max_completion_tokens=max_completion_tokens,
                format_max_attempts=format_max_attempts,
            )
        except Exception as error:
            raise RoleScreeningTechnicalError(
                f"Role check failed: {error}",
                audit={
                    "status": "technical_failure",
                    "edit_index": edit_index,
                    "skill_index": skill_index,
                    "error": str(error),
                },
            ) from error

        passed = result.get("passed")
        status = result.get("status")

        valid_pass = (
            passed is True
            and status in _PASS
            and structured_atomic_skill_error(
                result.get("candidate_skill")
            ) is None
        )
        valid_drop = passed is False and status in _DROP

        if not valid_pass and not valid_drop:
            raise RoleScreeningTechnicalError(
                "No usable role-consistency decision.",
                audit={
                    "status": "invalid_role_result",
                    "edit_index": edit_index,
                    "skill_index": skill_index,
                    "result": result,
                },
            )

        return edit_index, skill_index, result

    decisions = {}
    if items:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(items))
        ) as pool:
            for edit_index, skill_index, result in pool.map(
                check_one, items
            ):
                decisions[(edit_index, skill_index)] = result

    filtered = copy.deepcopy(canonical)
    edits = []
    records = []
    output_ids = set()

    for edit_index, edit in enumerate(canonical["edits"]):
        if edit["op"] == "delete":
            edits.append(copy.deepcopy(edit))
            continue

        kept = []
        kept_ids = set()

        for skill_index, skill in enumerate(edit["skills"]):
            result = decisions[(edit_index, skill_index)]
            record = {
                "edit_index": edit_index,
                "skill_index": skill_index,
                "source_skill_id": skill_id_from_draft(skill),
                "role_consistency": result,
            }

            if not result["passed"]:
                record["action"] = "dropped"
                records.append(record)
                continue

            approved = copy.deepcopy(result["candidate_skill"])
            output_id = skill_id_from_draft(approved)
            record["output_skill_id"] = output_id

            # Two repairs may converge to exactly the same two-field skill.
            if output_id in kept_ids:
                record["action"] = "duplicate_exact_output"
            else:
                kept.append(approved)
                kept_ids.add(output_id)
                output_ids.add(output_id)
                record["action"] = (
                    "passed" if approved == skill else "repaired"
                )
            records.append(record)

        if kept:
            retained = copy.deepcopy(edit)
            retained["skills"] = kept
            edits.append(retained)

    filtered["edits"] = edits

    if "_evidence_sidecar" in canonical:
        sidecar = canonical["_evidence_sidecar"]
        filtered["_evidence_sidecar"] = {
            skill_id: copy.deepcopy(sidecar[skill_id])
            for skill_id in sorted(output_ids)
            if skill_id in sidecar
        }

    try:
        filtered = parse_patch_payload(filtered, **parse_options)
    except Exception as error:
        raise RoleScreeningTechnicalError(
            f"Invalid role-screened output: {error}",
            audit={
                "status": "invalid_filtered_patch",
                "error": str(error),
                "results": records,
            },
        ) from error

    return filtered, {
        "status": "pass" if edits else "semantic_empty",
        "input_proposal_count": len(canonical["edits"]),
        "output_proposal_count": len(edits),
        "input_atomic_skill_count": len(items),
        "output_atomic_skill_count": sum(
            len(edit.get("skills", [])) for edit in edits
        ),
        "dropped_count": sum(
            row["action"] == "dropped" for row in records
        ),
        "repaired_count": sum(
            row["action"] == "repaired" for row in records
        ),
        "reused_role_approval_count": sum(
            row["role_consistency"].get(
                "role_approval_cache", {}
            ).get("hit") is True
            for row in records
        ),
        "results": records,
    }
