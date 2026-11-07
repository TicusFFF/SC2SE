"""Helpers for switching between patch edits and rewrite-from-suggestions."""
from __future__ import annotations

import copy
from typing import Any

from skillopt.optimizer.structured_skills import (
    canonical_json_text,
    compute_skill_id,
    parse_patch_edit,
    parse_patch_payload,
)

PATCH_MODE = "patch"
REWRITE_MODE = "rewrite_from_suggestions"
FULL_REWRITE_MINIBATCH_MODE = "full_rewrite_minibatch"


def normalize_update_mode(mode: str | None) -> str:
    raw = str(mode or PATCH_MODE).strip().lower()
    aliases = {
        "patch": PATCH_MODE,
        "edits": PATCH_MODE,
        "rewrite": REWRITE_MODE,
        "rewrite_from_suggestions": REWRITE_MODE,
        "suggestions": REWRITE_MODE,
        "rewrite_suggestions": REWRITE_MODE,
        "full_rewrite": FULL_REWRITE_MINIBATCH_MODE,
        "full_rewrite_minibatch": FULL_REWRITE_MINIBATCH_MODE,
        "minibatch_full_rewrite": FULL_REWRITE_MINIBATCH_MODE,
        "skill_rewrite_minibatch": FULL_REWRITE_MINIBATCH_MODE,
    }
    return aliases.get(raw, PATCH_MODE)


def is_rewrite_mode(mode: str | None) -> bool:
    return normalize_update_mode(mode) == REWRITE_MODE


def is_full_rewrite_minibatch_mode(mode: str | None) -> bool:
    return normalize_update_mode(mode) == FULL_REWRITE_MINIBATCH_MODE


def payload_key(mode: str | None) -> str:
    if is_full_rewrite_minibatch_mode(mode):
        return "skill_candidates"
    return "revise_suggestions" if is_rewrite_mode(mode) else "edits"


def payload_label(mode: str | None, *, singular: bool = False, title: bool = False) -> str:
    if is_full_rewrite_minibatch_mode(mode):
        word = "skill candidate" if singular else "skill candidates"
    elif is_rewrite_mode(mode):
        word = "suggestion" if singular else "suggestions"
    else:
        word = "edit" if singular else "edits"
    return word.title() if title else word


def get_payload_items(container: dict | None, mode: str | None) -> list[dict]:
    if not isinstance(container, dict):
        return []
    items = container.get(payload_key(mode), [])
    return items if isinstance(items, list) else []


def set_payload_items(container: dict, items: list[dict], mode: str | None) -> dict:
    container[payload_key(mode)] = items
    return container


def truncate_payload(container: dict, max_items: int, mode: str | None) -> dict:
    if max_items < 0:
        return container
    items = get_payload_items(container, mode)
    if len(items) > max_items:
        set_payload_items(container, items[:max_items], mode)
    return container


def canonicalize_patch_edit(
    item: dict,
) -> dict:
    """Validate one JSON Edit without accepting Markdown/text fallbacks."""

    return parse_patch_edit(item, path="$.edit")


def canonicalize_patch(
    patch: dict,
) -> dict:
    """Return a canonical trusted-internal structured JSON patch."""

    return parse_patch_payload(
        copy.deepcopy(patch),
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def describe_item(
    item: dict,
    mode: str | None,
    *,
    max_chars: int | None = None,
) -> str:
    """Describe one complete update item without discarding content.

    ``max_chars`` is retained only for call compatibility and is
    intentionally ignored.
    """
    del max_chars

    if not isinstance(item, dict):
        return ""

    if is_full_rewrite_minibatch_mode(mode):
        parts = [
            f"title={item.get('title', '')!r}",
            f"change_summary={item.get('change_summary', [])!r}",
        ]
        if item.get("source_type"):
            parts.append(f"source={item.get('source_type')}")
        if item.get("support_count") is not None:
            parts.append(f"support={item.get('support_count')}")
        new_skill = str(item.get("new_skill", "")).strip()
        if new_skill:
            parts.append(f"new_skill={new_skill!r}")
        return "  ".join(parts)

    if is_rewrite_mode(mode):
        parts = [
            f"type={item.get('type', '?')}",
            f"title={item.get('title', '')!r}",
            f"instruction={item.get('instruction', '')!r}",
        ]
        if item.get("priority_hint"):
            parts.append(f"priority={item.get('priority_hint')}")
        if item.get("support_count") is not None:
            parts.append(f"support={item.get('support_count')}")
        return "  ".join(parts)

    canonical = canonicalize_patch_edit(item)
    parts = [f"op={canonical.get('op', '?')}"]
    target_skill_id = canonical.get("target_skill_id", "")
    if target_skill_id:
        parts.append(f"target_skill_id={target_skill_id!r}")
    skills = canonical.get("skills", [])
    if skills:
        parts.append(
            "skills="
            + canonical_json_text(
                [
                    {
                        "skill_id": compute_skill_id(skill),
                        **skill,
                    }
                    for skill in skills
                ]
            )
        )
    if canonical.get("support_count") is not None:
        parts.append(f"support={canonical.get('support_count')}")
    return "  ".join(parts)


def short_item_summary(
    item: dict,
    mode: str | None,
    *,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Keep the complete fields needed by the rejected-update buffer."""
    del max_chars

    if not isinstance(item, dict):
        return {}

    if is_full_rewrite_minibatch_mode(mode):
        return {
            "title": str(item.get("title", "")),
            "change_summary": [
                str(value)
                for value in item.get("change_summary", [])
            ] if isinstance(item.get("change_summary"), list) else [],
            "source_type": item.get("source_type", ""),
            "new_skill": str(item.get("new_skill", "")),
        }
    if is_rewrite_mode(mode):
        return {
            "type": item.get("type", "?"),
            "title": str(item.get("title", "")),
            "instruction": str(item.get("instruction", "")),
        }
    canonical = canonicalize_patch_edit(item)
    summary: dict[str, Any] = {
        "op": canonical["op"],
    }
    if "target_skill_id" in canonical:
        summary["target_skill_id"] = canonical[
            "target_skill_id"
        ]
    if "skills" in canonical:
        summary["skills"] = copy.deepcopy(
            canonical["skills"]
        )
    return summary
