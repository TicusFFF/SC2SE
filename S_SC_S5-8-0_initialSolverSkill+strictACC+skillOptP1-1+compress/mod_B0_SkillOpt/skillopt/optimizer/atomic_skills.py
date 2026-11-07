"""Atomic JSON skill views over transport-only parent Edits.

Every element of an Edit's ``skills`` array is one atomic solver skill.
Markdown headings and textual matching are deliberately absent.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from skillopt.optimizer.structured_skills import (
    PATCH_SCHEMA_VERSION,
    canonical_json_bytes,
    compute_skill_id,
)
from skillopt.optimizer.update_modes import (
    canonicalize_patch,
    canonicalize_patch_edit,
)


SCHEMA_VERSION = "atomic_structured_skill_view_v1"


def _delete_atomic_id(target_skill_id: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes({
            "op": "delete",
            "target_skill_id": target_skill_id,
        })
    ).hexdigest()
    return "delete_sha256_" + digest


def build_atomic_skill_view(patch: dict) -> dict[str, Any]:
    """Expand structured parent Edits into independently rankable atoms."""

    canonical_patch = canonicalize_patch(patch)
    parents: list[dict[str, Any]] = []
    atomic_skills: list[dict[str, Any]] = []
    seen_atomic_ids: set[str] = set()

    for parent_edit_index, edit in enumerate(canonical_patch["edits"]):
        op = str(edit["op"])
        target_skill_id = str(edit.get("target_skill_id") or "")
        parent_ids: list[str] = []

        if op == "delete":
            atomic_skill_id = _delete_atomic_id(target_skill_id)
            if atomic_skill_id in seen_atomic_ids:
                raise ValueError(
                    f"Duplicate atomic operation id: {atomic_skill_id}"
                )
            seen_atomic_ids.add(atomic_skill_id)
            parent_ids.append(atomic_skill_id)
            atomic_skills.append({
                "atomic_skill_id": atomic_skill_id,
                "parent_edit_index": parent_edit_index,
                "skill_index": 0,
                "unit_type": "delete",
                "op": op,
                "target_skill_id": target_skill_id,
            })
        else:
            for skill_index, skill in enumerate(edit["skills"]):
                skill_id = compute_skill_id(skill)
                if skill_id in seen_atomic_ids:
                    raise ValueError(
                        "A structured patch must not repeat an atomic "
                        f"skill: {skill_id}"
                    )
                seen_atomic_ids.add(skill_id)
                parent_ids.append(skill_id)
                atom: dict[str, Any] = {
                    "atomic_skill_id": skill_id,
                    "skill_id": skill_id,
                    "parent_edit_index": parent_edit_index,
                    "skill_index": skill_index,
                    "unit_type": "skill",
                    "op": op,
                    "skill": copy.deepcopy(skill),
                }
                if target_skill_id:
                    atom["target_skill_id"] = target_skill_id
                atomic_skills.append(atom)

        parents.append({
            "parent_edit_index": parent_edit_index,
            "edit": copy.deepcopy(edit),
            "atomic_skill_ids": parent_ids,
        })

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_reasoning": str(canonical_patch.get("reasoning") or ""),
        "parents": parents,
        "atomic_skills": atomic_skills,
    }
    if "_evidence_sidecar" in canonical_patch:
        result["_evidence_sidecar"] = copy.deepcopy(
            canonical_patch["_evidence_sidecar"]
        )
    return result


def rebuild_patch_from_atomic_selection(
    view: dict[str, Any],
    selected_atomic_skill_ids: list[str],
    *,
    reasoning: str,
    ranking_details: dict[str, Any],
) -> dict:
    """Map selected structured atoms back to their parent Edit budget."""

    item_by_id = {
        str(item["atomic_skill_id"]): item
        for item in view["atomic_skills"]
    }
    if len(selected_atomic_skill_ids) != len(set(selected_atomic_skill_ids)):
        raise ValueError("Selected atomic skill ids must be unique.")
    unknown_ids = [
        item_id
        for item_id in selected_atomic_skill_ids
        if item_id not in item_by_id
    ]
    if unknown_ids:
        raise ValueError(f"Unknown atomic skill ids: {unknown_ids}")

    priority = {
        item_id: index
        for index, item_id in enumerate(selected_atomic_skill_ids)
    }
    selected = set(selected_atomic_skill_ids)
    selected_parents = [
        parent
        for parent in view["parents"]
        if any(item_id in selected for item_id in parent["atomic_skill_ids"])
    ]
    selected_parents.sort(
        key=lambda parent: (
            min(
                priority[item_id]
                for item_id in parent["atomic_skill_ids"]
                if item_id in selected
            ),
            int(parent["parent_edit_index"]),
        )
    )

    rebuilt_edits: list[dict[str, Any]] = []
    selected_skill_ids: set[str] = set()
    for parent in selected_parents:
        edit = copy.deepcopy(parent["edit"])
        if edit["op"] == "delete":
            rebuilt_edits.append(canonicalize_patch_edit(edit))
            continue

        kept_skills: list[dict[str, str]] = []
        for skill in edit["skills"]:
            skill_id = compute_skill_id(skill)
            if skill_id in selected:
                kept_skills.append(copy.deepcopy(skill))
                selected_skill_ids.add(skill_id)
        if not kept_skills:
            continue
        edit["skills"] = kept_skills
        rebuilt_edits.append(canonicalize_patch_edit(edit))

    rebuilt: dict[str, Any] = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "reasoning": reasoning,
        "edits": rebuilt_edits,
        "ranking_details": copy.deepcopy(ranking_details),
    }
    sidecar = view.get("_evidence_sidecar")
    if isinstance(sidecar, dict):
        rebuilt["_evidence_sidecar"] = {
            skill_id: copy.deepcopy(sidecar[skill_id])
            for skill_id in sorted(selected_skill_ids)
            if skill_id in sidecar
        }
    return canonicalize_patch(rebuilt)


def infer_selected_atomic_skill_ids(
    view: dict[str, Any],
    selected_patch: dict,
) -> list[str]:
    """Read selected structured IDs without any text-signature fallback."""

    selected_view = build_atomic_skill_view(selected_patch)
    known_ids = {
        str(item["atomic_skill_id"])
        for item in view["atomic_skills"]
    }
    selected_ids = [
        str(item["atomic_skill_id"])
        for item in selected_view["atomic_skills"]
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected atomic skill ids must be unique.")
    unknown = [item_id for item_id in selected_ids if item_id not in known_ids]
    if unknown:
        raise ValueError(
            "The selected patch contains unknown structured atomic ids: "
            f"{unknown}"
        )
    return selected_ids


def atomic_skill_view_artifact(
    view: dict[str, Any],
    selected_atomic_skill_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the auditable public part of a structured atomic view."""

    selected = (
        set(selected_atomic_skill_ids)
        if selected_atomic_skill_ids is not None
        else None
    )
    if selected is not None:
        known = {
            str(item["atomic_skill_id"])
            for item in view["atomic_skills"]
        }
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown selected atomic skill ids: {unknown}")

    items: list[dict[str, Any]] = []
    for item in view["atomic_skills"]:
        record = copy.deepcopy(item)
        if selected is not None:
            record["selected"] = item["atomic_skill_id"] in selected
        items.append(record)

    return {
        "schema_version": view["schema_version"],
        "parent_edit_count": len(view["parents"]),
        "atomic_skill_count": len(items),
        "selected_atomic_skill_count": (
            None
            if selected is None
            else sum(
                item["atomic_skill_id"] in selected
                for item in view["atomic_skills"]
            )
        ),
        "atomic_skills": items,
    }


__all__ = [
    "SCHEMA_VERSION",
    "atomic_skill_view_artifact",
    "build_atomic_skill_view",
    "infer_selected_atomic_skill_ids",
    "rebuild_patch_from_atomic_selection",
]
