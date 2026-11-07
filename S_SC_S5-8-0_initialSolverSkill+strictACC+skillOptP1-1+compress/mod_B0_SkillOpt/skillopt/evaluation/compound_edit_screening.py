"""Screen atomic Skills inside structured or legacy compound Edits.

Canonical JSON Edits are screened directly through ``edits[].skills[]``.
The Markdown split/rebuild path remains only for legacy ``content`` Edits.
Surviving atomic Skills stay in the same parent Edit before Aggregate,
Rank, Apply, and whole-candidate ACC stages.
"""
from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from skillopt.evaluation.role_consistency import (
    enforce_ranked_patch_role_consistency,
    enforce_structured_atomic_skill_role_consistency,
    structured_atomic_skill_error,
)
from skillopt.optimizer.structured_skills import (
    compute_skill_id,
)


SCHEMA_VERSION = "preaggregate_compound_screening_v4"

_HEADING_RE = re.compile(
    r"^ {0,3}(#{1,6})(?:[ \t]+|$)"
)


def _sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _heading_positions(
    content: str,
) -> tuple[list[dict[str, int]], bool]:
    """Return headings outside fenced code and whether a fence is unclosed."""
    headings: list[dict[str, int]] = []

    fence_char: str | None = None
    fence_length = 0
    offset = 0

    for line in content.splitlines(
        keepends=True
    ):
        body = line.rstrip("\r\n")
        leading = len(body) - len(
            body.lstrip(" ")
        )
        candidate = body[leading:]

        if fence_char is not None:
            if (
                leading <= 3
                and candidate.startswith(
                    fence_char * fence_length
                )
            ):
                run_length = (
                    len(candidate)
                    - len(
                        candidate.lstrip(
                            fence_char
                        )
                    )
                )
                remainder = candidate[
                    run_length:
                ]

                if (
                    run_length >= fence_length
                    and not remainder.strip()
                ):
                    fence_char = None
                    fence_length = 0

            offset += len(line)
            continue

        if (
            leading <= 3
            and candidate
            and candidate[0] in {
                "`",
                "~",
            }
        ):
            possible_char = candidate[0]
            run_length = (
                len(candidate)
                - len(
                    candidate.lstrip(
                        possible_char
                    )
                )
            )

            if run_length >= 3:
                fence_char = possible_char
                fence_length = run_length
                offset += len(line)
                continue

        heading_match = _HEADING_RE.match(
            body
        )

        if heading_match is not None:
            headings.append({
                "start": offset,
                "level": len(
                    heading_match.group(1)
                ),
            })

        offset += len(line)

    return headings, fence_char is not None


def _is_scaffold(
    text: str,
) -> bool:
    """A scaffold may contain only whitespace and Markdown headings."""
    for line in text.splitlines():
        if not line.strip():
            continue

        if _HEADING_RE.match(
            line
        ) is None:
            return False

    return True


def _terminal_separator(
    text: str,
) -> str:
    match = re.search(
        r"[ \t\r\n]*\Z",
        text,
    )
    return (
        match.group(0)
        if match is not None
        else ""
    )


def _whole_edit_plan(
    content: str,
    *,
    reason: str,
) -> dict[str, Any]:
    separator = _terminal_separator(
        content
    )
    core_text = (
        content[
            :len(content) - len(separator)
        ]
        if separator
        else content
    )

    return {
        "mode": "whole_edit",
        "fallback_reason": reason,
        "prefix": "",
        "parts": [{
            "part_index": 0,
            "char_start": 0,
            "char_end": len(content),
            "byte_start": 0,
            "byte_end": len(
                content.encode("utf-8")
            ),
            "raw_text": content,
            "core_text": core_text,
            "separator": separator,
            "scaffold_after": "",
        }],
    }


def split_compound_content(
    content: str,
) -> dict[str, Any]:
    """Deterministically split one legacy compound Edit by heading structure.

    The function never asks an LLM to choose boundaries.
    """
    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        raise ValueError(
            "Compound Edit content must be a non-empty string."
        )

    headings, unclosed_fence = (
        _heading_positions(content)
    )

    if unclosed_fence:
        raise ValueError(
            "Compound Edit contains an unclosed fenced code block; "
            "refusing unsafe whole-edit role screening."
        )

    has_explicit_h3 = any(
        heading["level"] == 3
        for heading in headings
    )

    if (
        len(headings) < 2
        and not has_explicit_h3
    ):
        return _whole_edit_plan(
            content,
            reason="fewer_than_two_headings",
        )

    counts_by_level: dict[int, int] = {}
    for heading in headings:
        level = heading["level"]
        counts_by_level[level] = (
            counts_by_level.get(
                level,
                0,
            )
            + 1
        )

    # S5-7-5's Analyst contract uses level-3 headings as the
    # explicit child-subskill boundary. If any H3 heading is
    # present, do not descend into H4/H5/H6 subsections.
    if counts_by_level.get(3, 0) > 0:
        split_level = 3

    else:
        # Backward-compatible fallback for content that contains
        # no H3 headings.
        shallowest_level = min(
            counts_by_level
        )

        if (
            counts_by_level[
                shallowest_level
            ]
            >= 2
        ):
            split_level = shallowest_level

        else:
            repeated_deeper_levels = [
                level
                for level, count
                in counts_by_level.items()
                if (
                    level > shallowest_level
                    and count >= 2
                )
            ]

            if not repeated_deeper_levels:
                return _whole_edit_plan(
                    content,
                    reason=(
                        "no_repeated_reviewable_heading_level"
                    ),
                )

            split_level = min(
                repeated_deeper_levels
            )

    boundaries = [
        heading
        for heading in headings
        if heading["level"] == split_level
    ]

    # One explicit H3 is already one complete child. H4/H5/H6
    # subsections remain inside it and are not split again.
    if (
        len(boundaries) < 2
        and split_level != 3
    ):
        return _whole_edit_plan(
            content,
            reason="fewer_than_two_boundaries",
        )

    prefix = content[
        :boundaries[0]["start"]
    ]

    if (
        split_level != 3
        and not _is_scaffold(prefix)
    ):
        return _whole_edit_plan(
            content,
            reason=(
                "substantive_text_before_first_boundary"
            ),
        )

    parts: list[dict[str, Any]] = []

    for index, boundary in enumerate(
        boundaries
    ):
        start = boundary["start"]
        span_end = (
            boundaries[index + 1]["start"]
            if index + 1 < len(boundaries)
            else len(content)
        )

        # A shallower heading closes the current child and starts
        # immutable parent scaffold. Keep that scaffold out of the
        # role-consistency request and restore it deterministically.
        scaffold_start = next(
            (
                heading["start"]
                for heading in headings
                if (
                    start < heading["start"]
                    and heading["start"] < span_end
                    and heading["level"] < split_level
                )
            ),
            span_end,
        )

        raw_text = content[
            start:scaffold_start
        ]
        scaffold_after = content[
            scaffold_start:span_end
        ]

        if not raw_text.strip():
            return _whole_edit_plan(
                content,
                reason="empty_heading_section",
            )

        separator = _terminal_separator(
            raw_text
        )
        core_text = (
            raw_text[
                :len(raw_text) - len(separator)
            ]
            if separator
            else raw_text
        )

        parts.append({
            "part_index": index,
            "char_start": start,
            "char_end": scaffold_start,
            "span_char_end": span_end,
            "byte_start": len(
                content[:start].encode(
                    "utf-8"
                )
            ),
            "byte_end": len(
                content[:scaffold_start].encode(
                    "utf-8"
                )
            ),
            "span_byte_end": len(
                content[:span_end].encode(
                    "utf-8"
                )
            ),
            "raw_text": raw_text,
            "core_text": core_text,
            "separator": separator,
            "scaffold_after": scaffold_after,
        })

    reconstructed = (
        prefix
        + "".join(
            (
                part["raw_text"]
                + part["scaffold_after"]
            )
            for part in parts
        )
    )

    if reconstructed != content:
        raise AssertionError(
            "Compound Edit split did not round-trip as text."
        )

    if (
        reconstructed.encode("utf-8")
        != content.encode("utf-8")
    ):
        raise AssertionError(
            "Compound Edit split did not round-trip as UTF-8 bytes."
        )

    return {
        "mode": "heading_sections",
        "fallback_reason": None,
        "split_level": split_level,
        "prefix": prefix,
        "parts": parts,
    }


def rebuild_compound_content(
    original_content: str,
    split_plan: dict[str, Any],
    selected_core_by_part_index: dict[int, str],
) -> str:
    """Rebuild selected H3 parts while preserving parent scaffold and bytes."""
    parts = split_plan.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Split plan must contain a parts list.")

    valid_indices = {
        int(part["part_index"])
        for part in parts
    }
    unknown_indices = (
        set(selected_core_by_part_index)
        - valid_indices
    )
    if unknown_indices:
        raise ValueError(
            "Unknown split part indices: "
            f"{sorted(unknown_indices)}"
        )

    rebuilt_parts: list[str] = [
        str(split_plan.get("prefix") or "")
    ]

    for part in parts:
        part_index = int(part["part_index"])

        if part_index in selected_core_by_part_index:
            final_core = selected_core_by_part_index[
                part_index
            ]
            if final_core == part["core_text"]:
                rebuilt_parts.append(part["raw_text"])
            else:
                rebuilt_parts.append(
                    _restore_separator(
                        final_core,
                        part["separator"],
                    )
                )

        rebuilt_parts.append(
            str(part.get("scaffold_after") or "")
        )

    return _restore_original_eof_newlines(
        "".join(rebuilt_parts),
        original_content,
    )


def _gate_record(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key != "filtered_patch"
    }


def _filtered_content_by_id(
    patch: dict[str, Any],
    *,
    expected_ids: list[str],
) -> tuple[dict[str, str], list[str]]:
    edits = patch.get("edits")

    if not isinstance(edits, list):
        raise ValueError(
            "Filtered synthetic patch must contain an edits list."
        )

    expected_position = {
        subskill_id: index
        for index, subskill_id
        in enumerate(expected_ids)
    }

    content_by_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    previous_position = -1

    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError(
                "Filtered synthetic edit must be a dictionary."
            )

        subskill_id = str(
            edit.get("subskill_id")
            or ""
        ).strip()

        if subskill_id not in expected_position:
            raise ValueError(
                "Filtered synthetic patch contains an unknown "
                f"subskill_id: {subskill_id!r}."
            )

        if subskill_id in content_by_id:
            raise ValueError(
                "Filtered synthetic patch contains a duplicate "
                f"subskill_id: {subskill_id!r}."
            )

        position = expected_position[
            subskill_id
        ]

        if position <= previous_position:
            raise ValueError(
                "Filtered synthetic subskills changed source order."
            )

        content = edit.get("content")
        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                "A surviving synthetic subskill must contain "
                "non-empty content."
            )

        content_by_id[subskill_id] = (
            content
        )
        ordered_ids.append(subskill_id)
        previous_position = position

    return content_by_id, ordered_ids


def _restore_separator(
    repaired_core: str,
    original_separator: str,
) -> str:
    return (
        repaired_core.rstrip("\r\n")
        + original_separator
    )


def _restore_original_eof_newlines(
    rebuilt: str,
    original: str,
) -> str:
    original_match = re.search(
        r"[\r\n]*\Z",
        original,
    )
    original_newlines = (
        original_match.group(0)
        if original_match is not None
        else ""
    )

    return (
        rebuilt.rstrip("\r\n")
        + original_newlines
    )


def _is_structured_skill_edit(
    edit: dict[str, Any],
) -> bool:
    """Return whether a non-delete Edit uses the canonical skills array."""
    return (
        isinstance(edit, dict)
        and edit.get("op") != "delete"
        and "skills" in edit
    )


def _screen_structured_skill_edit(
    edit: dict[str, Any],
    *,
    edit_index: int,
    env: str,
    max_completion_tokens: int,
    format_max_attempts: int,
) -> dict[str, Any]:
    """Screen each JSON atomic Skill without converting to Markdown."""
    op = edit.get("op")
    skills = edit.get("skills")
    if op not in {
        "append",
        "insert_after",
        "replace",
    }:
        return {
            "passed": False,
            "status": "invalid_parent_edit",
            "filtered_edit": None,
            "error": f"Unsupported structured parent Edit op: {op!r}.",
        }
    if not isinstance(skills, list) or not skills:
        return {
            "passed": False,
            "status": "invalid_parent_edit",
            "filtered_edit": None,
            "error": (
                "A structured non-delete Edit must contain a non-empty "
                "skills array."
            ),
        }

    role_summary: dict[str, Any] = {
        "input_count": len(skills),
        "processed_count": 0,
        "output_count": len(skills),
        "passed_without_repair_count": 0,
        "repaired_count": 0,
        "dropped_count": 0,
        "delete_passthrough_count": 0,
        "technical_failure_count": 0,
        "committed": False,
    }
    role_results: list[dict[str, Any]] = []

    if op == "replace" and len(skills) != 1:
        role_summary.update({
            "processed_count": len(skills),
            "output_count": 0,
            "dropped_count": len(skills),
            "committed": True,
        })
        return {
            "passed": True,
            "status": "all_skills_dropped_after_role",
            "changed": True,
            "filtered_edit": None,
            "split": None,
            "role": {
                "passed": True,
                "status": "invalid_replace_cardinality_dropped",
                "changed": True,
                "summary": role_summary,
                "results": [],
            },
        }

    filtered_skills: list[dict[str, Any]] = []
    for skill_index, skill in enumerate(skills):
        schema_error = structured_atomic_skill_error(skill)
        if schema_error is not None:
            role_summary["dropped_count"] += 1
            role_results.append({
                "index": skill_index,
                "edit_index": edit_index,
                "action": "dropped",
                "status": "invalid_atomic_skill_schema",
                "changed": True,
                "error": schema_error,
            })
            continue

        source_skill_id = compute_skill_id(skill)
        try:
            gate_result = (
                enforce_structured_atomic_skill_role_consistency(
                    copy.deepcopy(skill),
                    env=env,
                    max_completion_tokens=max_completion_tokens,
                    format_max_attempts=format_max_attempts,
                )
            )
        except Exception as error:  # noqa: BLE001
            role_summary.update({
                "processed_count": len(role_results) + 1,
                "technical_failure_count": 1,
                "output_count": len(skills),
                "committed": False,
            })
            role_results.append({
                "index": skill_index,
                "edit_index": edit_index,
                "action": "technical_failure",
                "status": "structured_role_exception",
                "changed": False,
                "error": f"{type(error).__name__}: {error}",
            })
            return {
                "passed": False,
                "status": "role_technical_failure",
                "filtered_edit": None,
                "split": None,
                "role": {
                    "passed": False,
                    "status": "structured_role_exception",
                    "changed": False,
                    "summary": role_summary,
                    "results": role_results,
                },
                "error": f"{type(error).__name__}: {error}",
            }

        status = str(gate_result.get("status") or "")
        passed = gate_result.get("passed")
        gate_record = {
            key: copy.deepcopy(value)
            for key, value in gate_result.items()
            if key != "candidate_skill"
        }
        record: dict[str, Any] = {
            "index": skill_index,
            "edit_index": edit_index,
            "source_skill_id": source_skill_id,
            "action": "",
            "status": status,
            "changed": False,
            "role_consistency": gate_record,
        }

        if passed is True and status in {
            "pass_without_repair",
            "pass_after_repair",
        }:
            screened_skill = gate_result.get("candidate_skill")
            screened_error = structured_atomic_skill_error(
                screened_skill
            )
            if screened_error is not None:
                role_summary.update({
                    "processed_count": len(role_results) + 1,
                    "technical_failure_count": 1,
                    "output_count": len(skills),
                    "committed": False,
                })
                record.update({
                    "action": "technical_failure",
                    "status": "invalid_structured_role_result",
                    "error": screened_error,
                })
                role_results.append(record)
                return {
                    "passed": False,
                    "status": "role_technical_failure",
                    "filtered_edit": None,
                    "split": None,
                    "role": {
                        "passed": False,
                        "status": "invalid_structured_role_result",
                        "changed": False,
                        "summary": role_summary,
                        "results": role_results,
                    },
                    "error": screened_error,
                }

            screened_skill = copy.deepcopy(screened_skill)
            record["output_skill_id"] = compute_skill_id(
                screened_skill
            )
            filtered_skills.append(screened_skill)
            if screened_skill == skill:
                record["action"] = "passed_without_repair"
                role_summary[
                    "passed_without_repair_count"
                ] += 1
            else:
                record["action"] = "repaired"
                record["changed"] = True
                role_summary["repaired_count"] += 1
            role_results.append(record)
            continue

        if passed is False and status in {
            "fail_after_repair",
            "unrepairable_no_commit",
        }:
            record["action"] = "dropped"
            record["changed"] = True
            role_summary["dropped_count"] += 1
            role_results.append(record)
            continue

        role_summary.update({
            "processed_count": len(role_results) + 1,
            "technical_failure_count": 1,
            "output_count": len(skills),
            "committed": False,
        })
        record["action"] = "technical_failure"
        role_results.append(record)
        return {
            "passed": False,
            "status": "role_technical_failure",
            "filtered_edit": None,
            "split": None,
            "role": {
                "passed": False,
                "status": "structured_role_technical_failure",
                "changed": False,
                "summary": role_summary,
                "results": role_results,
            },
            "error": (
                "Structured role processing failed closed for "
                f"skills[{skill_index}] with status {status!r}."
            ),
        }

    role_summary.update({
        "processed_count": len(role_results),
        "output_count": len(filtered_skills),
        "committed": True,
    })
    role_changed = filtered_skills != skills
    role_record = {
        "passed": True,
        "status": (
            "pass_with_changes"
            if role_changed
            else "pass_without_changes"
        ),
        "changed": role_changed,
        "summary": role_summary,
        "results": role_results,
    }

    if not filtered_skills:
        return {
            "passed": True,
            "status": "all_skills_dropped_after_role",
            "changed": True,
            "filtered_edit": None,
            "split": None,
            "role": role_record,
        }

    filtered_edit = copy.deepcopy(edit)
    filtered_edit["skills"] = filtered_skills
    changed = filtered_edit != edit
    return {
        "passed": True,
        "status": (
            "pass_with_changes"
            if changed
            else "pass_without_changes"
        ),
        "changed": changed,
        "filtered_edit": filtered_edit,
        "split": None,
        "role": role_record,
        "dropped_skill_indices": [
            record["index"]
            for record in role_results
            if record.get("action") == "dropped"
        ],
    }


def screen_compound_edit(
    edit: dict[str, Any],
    *,
    edit_index: int,
    source_type: str,
    evidence_ids: list[str],
    evidence_rows: list[dict[str, Any]],
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
    full_candidate_review: bool = False,
) -> dict[str, Any]:
    """Screen one original parent Edit and rebuild at the same parent level."""
    if not isinstance(edit, dict):
        return {
            "passed": False,
            "status": "invalid_parent_edit",
            "filtered_edit": None,
            "error": "Parent Edit must be a dictionary.",
        }

    op = edit.get("op")

    if op == "delete":
        return {
            "passed": True,
            "status": "delete_passthrough",
            "changed": False,
            "filtered_edit": copy.deepcopy(
                edit
            ),
            "split": None,
            "role": None,
        }

    if _is_structured_skill_edit(edit):
        return _screen_structured_skill_edit(
            edit,
            edit_index=edit_index,
            env=env,
            max_completion_tokens=max_completion_tokens,
            format_max_attempts=format_max_attempts,
        )

    if op not in {
        "append",
        "insert_after",
        "replace",
    }:
        return {
            "passed": False,
            "status": "invalid_parent_edit",
            "filtered_edit": None,
            "error": (
                f"Unsupported parent Edit op: {op!r}."
            ),
        }

    content = edit.get("content")
    if (
        not isinstance(content, str)
        or not content.strip()
    ):
        return {
            "passed": False,
            "status": "invalid_parent_edit",
            "filtered_edit": None,
            "error": (
                "A non-delete parent Edit must contain "
                "non-empty content."
            ),
        }

    try:
        split_plan = split_compound_content(
            content
        )
    except Exception as error:  # noqa: BLE001
        return {
            "passed": False,
            "status": "split_failure",
            "filtered_edit": None,
            "error": (
                f"{type(error).__name__}: {error}"
            ),
        }

    part_ids = [
        f"edit_{edit_index:04d}_part_{part['part_index']:04d}"
        for part in split_plan["parts"]
    ]

    synthetic_patch = {
        "edits": [
            {
                "op": "append",
                "content": part["core_text"],
                "schema_version": SCHEMA_VERSION,
                "subskill_id": subskill_id,
                "evidence_ids": list(
                    evidence_ids
                ),
                "evidence_count": len(
                    evidence_ids
                ),
                "support_count": edit.get(
                    "support_count",
                    len(evidence_ids),
                ),
                "source_type": (
                    edit.get("source_type")
                    or source_type
                ),
            }
            for subskill_id, part
            in zip(
                part_ids,
                split_plan["parts"],
            )
        ]
    }

    role_result = (
        enforce_ranked_patch_role_consistency(
            synthetic_patch,
            env=env,
            max_completion_tokens=(
                max_completion_tokens
            ),
            format_max_attempts=(
                format_max_attempts
            ),
            full_candidate_review=(
                full_candidate_review
            ),
        )
    )

    if not role_result.get("passed"):
        return {
            "passed": False,
            "status": "role_technical_failure",
            "filtered_edit": None,
            "split": split_plan,
            "role": _gate_record(
                role_result
            ),
            "error": role_result.get(
                "error",
                "Role gate failed technically.",
            ),
        }

    role_filtered_patch = (
        role_result["filtered_patch"]
    )

    if not role_filtered_patch.get(
        "edits"
    ):
        return {
            "passed": True,
            "status": "all_parts_dropped_after_role",
            "changed": True,
            "filtered_edit": None,
            "split": split_plan,
            "role": _gate_record(
                role_result
            ),
        }

    final_content_by_id, final_ids = (
        _filtered_content_by_id(
            role_filtered_patch,
            expected_ids=part_ids,
        )
    )

    dropped_ids = [
        subskill_id
        for subskill_id in part_ids
        if subskill_id not in final_content_by_id
    ]

    if not final_ids:
        return {
            "passed": True,
            "status": "all_parts_dropped_after_role",
            "changed": True,
            "filtered_edit": None,
            "split": split_plan,
            "role": _gate_record(
                role_result
            ),
        }

    rebuilt_parts: list[str] = [
        split_plan["prefix"]
    ]

    for subskill_id, part in zip(
        part_ids,
        split_plan["parts"],
    ):
        if subskill_id in final_content_by_id:
            final_core = final_content_by_id[
                subskill_id
            ]

            if final_core == part["core_text"]:
                rebuilt_parts.append(
                    part["raw_text"]
                )
            else:
                rebuilt_parts.append(
                    _restore_separator(
                        final_core,
                        part["separator"],
                    )
                )

        # Parent scaffold is immutable and independent of whether
        # the preceding child survives.
        rebuilt_parts.append(
            part.get(
                "scaffold_after",
                "",
            )
        )

    rebuilt_content = (
        _restore_original_eof_newlines(
            "".join(rebuilt_parts),
            content,
        )
    )

    if not rebuilt_content.strip():
        return {
            "passed": True,
            "status": "empty_rebuilt_parent",
            "changed": True,
            "filtered_edit": None,
            "split": split_plan,
            "role": _gate_record(
                role_result
            ),
        }

    filtered_edit = copy.deepcopy(
        edit
    )
    filtered_edit["content"] = (
        rebuilt_content
    )

    changed = filtered_edit != edit

    if not changed:
        if rebuilt_content != content:
            raise AssertionError(
                "Unchanged parent Edit failed text identity."
            )

        if (
            rebuilt_content.encode("utf-8")
            != content.encode("utf-8")
        ):
            raise AssertionError(
                "Unchanged parent Edit failed byte identity."
            )

    return {
        "passed": True,
        "status": (
            "pass_with_changes"
            if changed
            else "pass_without_changes"
        ),
        "changed": changed,
        "filtered_edit": filtered_edit,
        "split": split_plan,
        "role": _gate_record(
            role_result
        ),
        "dropped_part_ids": dropped_ids,
    }


def _nonnegative_summary_count(
    summary: dict[str, Any],
    key: str,
) -> int:
    try:
        value = int(
            summary.get(key, 0)
            or 0
        )
    except (TypeError, ValueError):
        return 0

    return max(value, 0)


def _gate_summary(
    parent_result: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    gate_record = parent_result.get(
        stage
    )
    if not isinstance(gate_record, dict):
        return {}

    summary = gate_record.get(
        "summary"
    )
    if not isinstance(summary, dict):
        return {}

    return summary


def _summarize_child_screening(
    parent_results: list[dict[str, Any]],
    *,
    committed: bool,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {
        "input_child_subskill_count": 0,
        "role_passed_without_repair_count": 0,
        "role_repaired_count": 0,
        "role_dropped_count": 0,
        "output_child_subskill_count": 0,
        "child_retention_rate": None,
    }

    for parent_result in parent_results:
        if not isinstance(
            parent_result,
            dict,
        ):
            continue

        role_summary = _gate_summary(
            parent_result,
            "role",
        )

        statistics[
            "input_child_subskill_count"
        ] += _nonnegative_summary_count(
            role_summary,
            "input_count",
        )
        statistics[
            "role_passed_without_repair_count"
        ] += _nonnegative_summary_count(
            role_summary,
            "passed_without_repair_count",
        )
        statistics[
            "role_repaired_count"
        ] += _nonnegative_summary_count(
            role_summary,
            "repaired_count",
        )
        statistics[
            "role_dropped_count"
        ] += _nonnegative_summary_count(
            role_summary,
            "dropped_count",
        )

        if (
            committed
            and role_summary.get(
                "committed"
            ) is True
        ):
            statistics[
                "output_child_subskill_count"
            ] += _nonnegative_summary_count(
                role_summary,
                "output_count",
            )

    input_count = statistics[
        "input_child_subskill_count"
    ]
    output_count = statistics[
        "output_child_subskill_count"
    ]

    if committed and input_count > 0:
        statistics[
            "child_retention_rate"
        ] = output_count / input_count

    return statistics


def _reconcile_postaggregate_evidence_sidecar(
    source_patch: dict[str, Any],
    filtered_patch: dict[str, Any],
    parent_results: list[dict[str, Any]],
) -> None:
    """Re-key trusted evidence after per-atom repair or removal."""
    source_sidecar = source_patch.get("_evidence_sidecar")
    if not isinstance(source_sidecar, dict):
        return

    source_id_by_output_id: dict[str, str] = {}
    for parent_result in parent_results:
        role = parent_result.get("role")
        if not isinstance(role, dict):
            continue
        role_results = role.get("results")
        if not isinstance(role_results, list):
            continue
        for role_result in role_results:
            if not isinstance(role_result, dict):
                continue
            source_skill_id = str(
                role_result.get("source_skill_id") or ""
            )
            output_skill_id = str(
                role_result.get("output_skill_id") or ""
            )
            if source_skill_id and output_skill_id:
                source_id_by_output_id[
                    output_skill_id
                ] = source_skill_id

    rebuilt_sidecar: dict[str, list[dict[str, str]]] = {}
    for edit in filtered_patch.get("edits", []):
        if not isinstance(edit, dict):
            continue
        skills = edit.get("skills")
        if not isinstance(skills, list):
            continue
        for skill in skills:
            try:
                output_skill_id = compute_skill_id(skill)
            except Exception:  # noqa: BLE001
                continue
            source_skill_id = source_id_by_output_id.get(
                output_skill_id,
                output_skill_id,
            )
            refs = source_sidecar.get(source_skill_id)
            if isinstance(refs, list):
                rebuilt_sidecar[output_skill_id] = copy.deepcopy(
                    refs
                )

    filtered_patch["_evidence_sidecar"] = rebuilt_sidecar


def screen_postaggregate_patch(
    merged_patch: dict[str, Any],
    *,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Screen every packaged solver subskill without repackaging it."""
    original_patch = copy.deepcopy(
        merged_patch
    )

    if not isinstance(merged_patch, dict):
        return {
            "passed": False,
            "status": "invalid_postaggregate_patch",
            "changed": False,
            "filtered_patch": original_patch,
            "summary": {
                "input_parent_edit_count": 0,
                "output_parent_edit_count": 0,
                "technical_parent_drop_count": 1,
                "semantic_parent_drop_count": 0,
                "delete_passthrough_count": 0,
                "input_child_subskill_count": 0,
                "role_passed_without_repair_count": 0,
                "role_repaired_count": 0,
                "role_dropped_count": 0,
                "output_child_subskill_count": 0,
                "child_retention_rate": None,
                "committed": False,
            },
            "results": [],
            "error": (
                "Post-Aggregate patch must be a dictionary."
            ),
        }

    edits = merged_patch.get("edits")
    if not isinstance(edits, list):
        return {
            "passed": False,
            "status": "invalid_postaggregate_patch",
            "changed": False,
            "filtered_patch": original_patch,
            "summary": {
                "input_parent_edit_count": 0,
                "output_parent_edit_count": 0,
                "technical_parent_drop_count": 1,
                "semantic_parent_drop_count": 0,
                "delete_passthrough_count": 0,
                "input_child_subskill_count": 0,
                "role_passed_without_repair_count": 0,
                "role_repaired_count": 0,
                "role_dropped_count": 0,
                "output_child_subskill_count": 0,
                "child_retention_rate": None,
                "committed": False,
            },
            "results": [],
            "error": (
                "Post-Aggregate patch.edits must be a list."
            ),
        }

    filtered_edits: list[dict[str, Any]] = []
    parent_results: list[dict[str, Any]] = []
    semantic_parent_drop_count = 0
    delete_passthrough_count = 0

    for edit_index, edit in enumerate(edits):
        parent_result = screen_compound_edit(
            edit,
            edit_index=edit_index,
            source_type="postaggregate",
            evidence_ids=[],
            evidence_rows=[],
            env=env,
            max_completion_tokens=(
                max_completion_tokens
            ),
            format_max_attempts=(
                format_max_attempts
            ),
            full_candidate_review=True,
        )
        parent_record = {
            key: copy.deepcopy(value)
            for key, value in parent_result.items()
            if key != "filtered_edit"
        }
        parent_record["edit_index"] = (
            edit_index
        )
        parent_results.append(parent_record)

        if not parent_result.get("passed"):
            child_statistics = (
                _summarize_child_screening(
                    parent_results,
                    committed=False,
                )
            )
            return {
                "passed": False,
                "status": "technical_failure",
                "changed": False,
                "filtered_patch": original_patch,
                "summary": {
                    "input_parent_edit_count": len(edits),
                    "output_parent_edit_count": len(edits),
                    "technical_parent_drop_count": 1,
                    "semantic_parent_drop_count": 0,
                    "delete_passthrough_count": 0,
                    **child_statistics,
                    "committed": False,
                },
                "results": parent_results,
                "error": str(
                    parent_result.get("error")
                    or (
                        "Post-Aggregate per-subskill "
                        "screening failed technically."
                    )
                ),
            }

        filtered_edit = parent_result.get(
            "filtered_edit"
        )
        if filtered_edit is None:
            semantic_parent_drop_count += 1
            continue

        if parent_result.get(
            "status"
        ) == "delete_passthrough":
            delete_passthrough_count += 1

        filtered_edits.append(filtered_edit)

    filtered_patch = copy.deepcopy(
        merged_patch
    )
    filtered_patch["edits"] = (
        filtered_edits
    )
    _reconcile_postaggregate_evidence_sidecar(
        merged_patch,
        filtered_patch,
        parent_results,
    )
    changed = filtered_patch != merged_patch
    child_statistics = (
        _summarize_child_screening(
            parent_results,
            committed=True,
        )
    )
    summary = {
        "input_parent_edit_count": len(edits),
        "output_parent_edit_count": len(
            filtered_edits
        ),
        "technical_parent_drop_count": 0,
        "semantic_parent_drop_count": (
            semantic_parent_drop_count
        ),
        "delete_passthrough_count": (
            delete_passthrough_count
        ),
        **child_statistics,
        "committed": True,
    }

    return {
        "passed": True,
        "status": (
            "semantic_empty"
            if not filtered_edits
            else (
                "pass_with_changes"
                if changed
                else "pass_without_changes"
            )
        ),
        "changed": changed,
        "filtered_patch": filtered_patch,
        "summary": summary,
        "results": parent_results,
    }


def _safe_empty_filtered_result(
    raw_result: dict[str, Any],
    *,
    status: str,
    error: str,
) -> dict[str, Any]:
    filtered_result = copy.deepcopy(
        raw_result
    )

    inner_patch = filtered_result.get(
        "patch"
    )
    if isinstance(inner_patch, dict):
        inner_patch["edits"] = []

    child_statistics = (
        _summarize_child_screening(
            [],
            committed=False,
        )
    )

    summary = {
        "input_parent_edit_count": 0,
        "output_parent_edit_count": 0,
        "technical_parent_drop_count": 0,
        "semantic_parent_drop_count": 0,
        "delete_passthrough_count": 0,
        **child_statistics,
        "cacheable": False,
    }

    filtered_result[
        "_preaggregate_compound_screening"
    ] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "status": status,
        "summary": summary,
        "error": error,
    }

    return {
        "passed": False,
        "status": status,
        "changed": True,
        "cacheable": False,
        "filtered_result": filtered_result,
        "summary": summary,
        "results": [],
        "error": error,
    }


def screen_reflect_result(
    raw_result: dict[str, Any],
    *,
    exact_minibatch_rows: list[dict[str, Any]],
    batch_tag: str,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
    format_max_attempts: int = 3,
) -> dict[str, Any]:
    """Screen every compound parent Edit produced by one Analyst minibatch."""
    if not isinstance(raw_result, dict):
        return _safe_empty_filtered_result(
            {},
            status="invalid_raw_result",
            error="Raw Reflect result must be a dictionary.",
        )

    inner_patch = raw_result.get(
        "patch"
    )
    if not isinstance(inner_patch, dict):
        return _safe_empty_filtered_result(
            raw_result,
            status="invalid_raw_patch",
            error="Raw Reflect result.patch must be a dictionary.",
        )

    edits = inner_patch.get(
        "edits"
    )
    if not isinstance(edits, list):
        return _safe_empty_filtered_result(
            raw_result,
            status="invalid_raw_patch",
            error="Raw Reflect patch.edits must be a list.",
        )

    evidence_ids: list[str] = []

    for index, row in enumerate(
        exact_minibatch_rows
    ):
        if not isinstance(row, dict):
            return _safe_empty_filtered_result(
                raw_result,
                status="invalid_evidence_rows",
                error=(
                    f"Evidence row {index} must be a dictionary."
                ),
            )

        evidence_id = str(
            row.get("id")
            or ""
        ).strip()

        if not evidence_id:
            return _safe_empty_filtered_result(
                raw_result,
                status="invalid_evidence_ids",
                error=(
                    f"Evidence row {index} has no non-empty id."
                ),
            )

        evidence_ids.append(
            evidence_id
        )

    if len(evidence_ids) != len(
        set(evidence_ids)
    ):
        return _safe_empty_filtered_result(
            raw_result,
            status="duplicate_evidence_ids",
            error="Exact minibatch evidence ids must be unique.",
        )

    filtered_result = copy.deepcopy(
        raw_result
    )
    filtered_edits: list[dict] = []
    parent_results: list[dict[str, Any]] = []

    technical_parent_drop_count = 0
    semantic_parent_drop_count = 0
    delete_passthrough_count = 0

    for edit_index, edit in enumerate(
        edits
    ):
        parent_result = screen_compound_edit(
            edit,
            edit_index=edit_index,
            source_type=str(
                raw_result.get(
                    "source_type"
                )
                or ""
            ),
            evidence_ids=evidence_ids,
            evidence_rows=(
                exact_minibatch_rows
            ),
            env=env,
            max_completion_tokens=(
                max_completion_tokens
            ),
            format_max_attempts=(
                format_max_attempts
            ),
        )

        parent_record = {
            key: copy.deepcopy(value)
            for key, value
            in parent_result.items()
            if key != "filtered_edit"
        }
        parent_record["edit_index"] = (
            edit_index
        )
        parent_results.append(
            parent_record
        )

        if not parent_result.get(
            "passed"
        ):
            technical_parent_drop_count += 1
            error = str(
                parent_result.get("error")
                or "Compound parent screening failed technically."
            )
            child_statistics = (
                _summarize_child_screening(
                    parent_results,
                    committed=False,
                )
            )
            summary = {
                "batch_tag": batch_tag,
                "evidence_ids": evidence_ids,
                "input_parent_edit_count": len(
                    edits
                ),
                "output_parent_edit_count": 0,
                "technical_parent_drop_count": (
                    technical_parent_drop_count
                ),
                "semantic_parent_drop_count": (
                    semantic_parent_drop_count
                ),
                "delete_passthrough_count": (
                    delete_passthrough_count
                ),
                **child_statistics,
                "cacheable": False,
            }

            # A model/API/format failure is not evidence that a child
            # subskill is semantically bad. Do not commit any partial
            # result from this minibatch.
            filtered_result["patch"]["edits"] = []
            filtered_result[
                "_preaggregate_compound_screening"
            ] = {
                "schema_version": SCHEMA_VERSION,
                "passed": False,
                "status": "technical_failure",
                "summary": summary,
                "error": error,
            }

            return {
                "passed": False,
                "status": "technical_failure",
                "changed": True,
                "cacheable": False,
                "filtered_result": filtered_result,
                "summary": summary,
                "results": parent_results,
                "error": error,
            }

        filtered_edit = parent_result.get(
            "filtered_edit"
        )

        if filtered_edit is None:
            semantic_parent_drop_count += 1
            continue

        if parent_result.get(
            "status"
        ) == "delete_passthrough":
            delete_passthrough_count += 1

        filtered_edits.append(
            filtered_edit
        )

    filtered_result["patch"]["edits"] = (
        filtered_edits
    )

    changed = filtered_edits != edits
    cacheable = True

    if filtered_edits:
        status = (
            "pass_with_changes"
            if changed
            else "pass_without_changes"
        )
    else:
        status = "semantic_empty"

    child_statistics = (
        _summarize_child_screening(
            parent_results,
            committed=True,
        )
    )

    summary = {
        "batch_tag": batch_tag,
        "evidence_ids": evidence_ids,
        "input_parent_edit_count": len(
            edits
        ),
        "output_parent_edit_count": len(
            filtered_edits
        ),
        "technical_parent_drop_count": (
            technical_parent_drop_count
        ),
        "semantic_parent_drop_count": (
            semantic_parent_drop_count
        ),
        "delete_passthrough_count": (
            delete_passthrough_count
        ),
        **child_statistics,
        "cacheable": cacheable,
    }

    filtered_result[
        "_preaggregate_compound_screening"
    ] = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "status": status,
        "summary": summary,
    }

    return {
        "passed": True,
        "status": status,
        "changed": changed,
        "cacheable": cacheable,
        "filtered_result": filtered_result,
        "summary": summary,
        "results": parent_results,
    }
