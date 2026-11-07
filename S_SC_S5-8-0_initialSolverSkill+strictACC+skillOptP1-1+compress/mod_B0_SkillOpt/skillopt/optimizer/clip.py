"""P1 proposal ranking with the S5 structured-skill transport format."""
from __future__ import annotations

import copy

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.structured_skills import (
    parse_patch_payload,
    skill_id_from_draft,
)
from skillopt.optimizer.update_modes import describe_item
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


def _select(patch, indices, details):
    selected = copy.deepcopy(patch)
    selected["edits"] = [
        copy.deepcopy(patch["edits"][index])
        for index in indices
    ]
    selected["ranking_details"] = details

    selected_ids = {
        skill_id_from_draft(skill)
        for edit in selected["edits"]
        for skill in edit.get("skills", [])
    }
    if "_evidence_sidecar" in patch:
        selected["_evidence_sidecar"] = {
            skill_id: copy.deepcopy(references)
            for skill_id, references in patch["_evidence_sidecar"].items()
            if skill_id in selected_ids
        }

    return parse_patch_payload(
        selected,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def rank_and_select(
    skill_content: str,
    patch: dict,
    max_edits: int,
    meta_skill_context: str = "",
    update_mode: str = "patch",
    max_completion_tokens: int = 0,
) -> dict:
    if update_mode != "patch":
        raise ValueError("S5 P1 migration supports patch mode only.")
    if type(max_edits) is not int or max_edits < 0:
        raise ValueError("max_edits must be a non-negative integer.")

    patch = parse_patch_payload(
        patch,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )
    edits = patch["edits"]

    if len(edits) <= max_edits:
        return patch
    if max_edits == 0:
        return _select(patch, [], {
            "status": "zero_budget",
            "unit": "edit_proposal",
        })

    descriptions = [
        f"[{index}] {describe_item(edit, update_mode)}"
        for index, edit in enumerate(edits)
    ]
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Edits Pool ({len(edits)} edits, budget={max_edits})\n"
        + "\n".join(descriptions)
        + f"\n\nSelect the {max_edits} most important edits. "
        "Return their 0-based indices in priority order."
    )
    context = format_meta_skill_context(meta_skill_context)
    if context:
        user = f"{context}\n\n{user}"

    for attempt in range(3):
        try:
            response, _ = chat_optimizer(
                system=load_prompt("ranking"),
                user=user,
                max_completion_tokens=max_completion_tokens,
                retries=4,
                stage="ranking",
            )
            result = extract_json(response)
            indices = result.get("selected_indices") if isinstance(
                result, dict
            ) else None
            if not isinstance(indices, list):
                raise ValueError("selected_indices must be a list.")

            selected = []
            seen = set()
            for index in indices:
                if (
                    type(index) is int
                    and 0 <= index < len(edits)
                    and index not in seen
                ):
                    selected.append(index)
                    seen.add(index)
                if len(selected) >= max_edits:
                    break

            if selected:
                return _select(patch, selected, {
                    **result,
                    "status": "ranked",
                    "unit": "edit_proposal",
                    "attempt_count": attempt + 1,
                })

            raise ValueError("Rank returned no usable indices.")

        except Exception as error:
            print(
                f"    [ranking] attempt {attempt + 1}/3 failed: "
                f"{type(error).__name__}: {error}"
            )

    return _select(
        patch,
        list(range(max_edits)),
        {
            "status": "fallback_input_order",
            "unit": "edit_proposal",
            "attempt_count": 3,
        },
    )
