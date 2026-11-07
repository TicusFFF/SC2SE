"""Deterministic ID-based updates for canonical JSON skill libraries."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from skillopt.optimizer.structured_skills import (
    PATCH_SCHEMA_VERSION,
    apply_structured_patch_with_report,
    dump_skill_library,
    parse_patch_payload,
    parse_skill_library,
)

if TYPE_CHECKING:
    from skillopt.types import Edit as EditType, Patch as PatchType


def _patch_to_dict(patch: PatchType | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(patch, "to_dict"):
        value = patch.to_dict()
    elif isinstance(patch, Mapping):
        value = dict(patch)
    else:
        raise TypeError("Patch must be a Patch instance or JSON object.")
    return parse_patch_payload(
        value,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def apply_edit(
    skill: str | Mapping[str, Any],
    edit: EditType | Mapping[str, Any],
) -> str:
    """Apply one structured Edit and return canonical library JSON text."""

    edit_value = edit.to_dict() if hasattr(edit, "to_dict") else dict(edit)
    patch = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "reasoning": "Single structured Edit application.",
        "edits": [edit_value],
    }
    updated, _ = apply_patch_with_report(skill, patch)
    return updated


def apply_patch_with_report(
    skill: str | Mapping[str, Any],
    patch: PatchType | Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply a structured patch transactionally and retain an audit report."""

    library = parse_skill_library(skill)
    canonical_patch = _patch_to_dict(patch)
    updated, reports = apply_structured_patch_with_report(
        library,
        canonical_patch,
    )
    return dump_skill_library(updated), reports


def apply_patch(
    skill: str | Mapping[str, Any],
    patch: PatchType | Mapping[str, Any],
) -> str:
    """Apply a structured patch and return canonical library JSON text."""

    updated, _ = apply_patch_with_report(skill, patch)
    return updated


__all__ = ["apply_edit", "apply_patch", "apply_patch_with_report"]
