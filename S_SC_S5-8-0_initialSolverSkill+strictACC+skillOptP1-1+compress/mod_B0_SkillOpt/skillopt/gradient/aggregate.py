"""P1 failure-first hierarchical merging for S5 solver-skill proposals."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.structured_skills import (
    PATCH_SCHEMA_VERSION,
    normalise_evidence,
    parse_patch_payload,
    skill_id_from_draft,
    strict_json_loads,
)
from skillopt.prompts import load_prompt


def _parse(patch):
    return parse_patch_payload(
        patch,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def _public_patch(patch):
    patch = _parse(patch)
    return {
        "schema_version": patch["schema_version"],
        "reasoning": patch["reasoning"],
        "edits": patch["edits"],
    }


def _attach_evidence(candidate, source_patches):
    candidate = parse_patch_payload(
        candidate,
        allow_ranking_details=False,
        allow_trusted_metadata=False,
    )

    evidence = {}
    for source in source_patches:
        for skill_id, references in _parse(source).get(
            "_evidence_sidecar", {}
        ).items():
            evidence[skill_id] = normalise_evidence([
                *evidence.get(skill_id, []),
                *references,
            ])

    output_ids = {
        skill_id_from_draft(skill)
        for edit in candidate["edits"]
        for skill in edit.get("skills", [])
    }

    candidate["_evidence_sidecar"] = {
        skill_id: evidence[skill_id]
        for skill_id in sorted(output_ids)
        if skill_id in evidence
    }
    return _parse(candidate)


def _fallback(patches, reason, level=None):
    edits = []
    for patch in patches:
        for item in _parse(patch)["edits"]:
            edit = copy.deepcopy(item)
            if level is not None:
                edit.setdefault("merge_level", level)
            edits.append(edit)

    return _attach_evidence(
        {
            "schema_version": PATCH_SCHEMA_VERSION,
            "reasoning": reason,
            "edits": edits,
        },
        patches,
    )


def _system(name):
    return "\n\n".join((
        load_prompt(name),
        load_prompt("role_consistency_role", env="forward_math"),
        load_prompt("merge_p1_format"),
    ))


def _call_merge(system, user, sources, max_completion_tokens):
    for attempt in range(3):
        retry_user = user
        if attempt:
            retry_user += (
                "\n\nReturn only one valid JSON object. "
                "Follow the output schema and field requirements "
                "in the system prompt."
            )

        try:
            response, _ = chat_optimizer(
                system=system,
                user=retry_user,
                max_completion_tokens=max_completion_tokens,
                retries=4,
                stage="merge",
            )
            return _attach_evidence(
                strict_json_loads(response),
                sources,
            )
        except Exception as error:
            print(
                f"      [merge] attempt {attempt + 1}/3 failed: "
                f"{type(error).__name__}: {error}"
            )

    return None


def _merge_batch(
    skill_content,
    patches,
    system_prompt,
    meta_skill_context,
    level,
    max_completion_tokens,
):
    patches_text = json.dumps(
        [_public_patch(patch) for patch in patches],
        ensure_ascii=False,
        indent=2,
    )
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Patches to merge "
        f"({len(patches)} total, merge level {level})\n"
        f"{patches_text}"
    )
    context = format_meta_skill_context(meta_skill_context)
    if context:
        user = f"{context}\n\n{user}"

    merged = _call_merge(
        system_prompt,
        user,
        patches,
        max_completion_tokens,
    )
    if merged is None:
        return _fallback(
            patches,
            "fallback concatenation",
            level,
        )

    for edit in merged["edits"]:
        edit["merge_level"] = level
    return _parse(merged)


def _hierarchical_merge(
    skill_content,
    patches,
    system_prompt,
    batch_size,
    workers,
    meta_skill_context,
    max_completion_tokens,
):
    if not patches:
        return {
            "schema_version": PATCH_SCHEMA_VERSION,
            "reasoning": "no patches",
            "edits": [],
        }

    current = [_parse(patch) for patch in patches]
    level = 0

    while len(current) > 1:
        level += 1
        batches = [
            current[index:index + batch_size]
            for index in range(0, len(current), batch_size)
        ]

        def merge_one(batch):
            if len(batch) == 1:
                return batch[0]
            return _merge_batch(
                skill_content,
                batch,
                system_prompt,
                meta_skill_context,
                level,
                max_completion_tokens,
            )

        with ThreadPoolExecutor(
            max_workers=min(workers, len(batches))
        ) as pool:
            current = list(pool.map(merge_one, batches))

    return current[0]


def merge_patches(
    skill_content: str,
    failure_patches: list[dict],
    success_patches: list[dict],
    batch_size: int = 8,
    verbose: bool = True,
    workers: int = 16,
    update_mode: str = "patch",
    meta_skill_context: str = "",
    max_completion_tokens: int = 0,
) -> dict:
    if update_mode != "patch":
        raise ValueError("S5 P1 migration supports patch mode only.")
    if batch_size < 2 or workers < 1:
        raise ValueError("Merge batch_size >= 2 and workers >= 1 required.")

    if verbose:
        print(
            f"    [3/6 AGGREGATE] failure={len(failure_patches)} "
            f"success={len(success_patches)} workers={workers}"
        )

    failure = _hierarchical_merge(
        skill_content,
        failure_patches,
        _system("merge_failure"),
        batch_size,
        workers,
        meta_skill_context,
        max_completion_tokens,
    )
    success = _hierarchical_merge(
        skill_content,
        success_patches,
        _system("merge_success"),
        batch_size,
        workers,
        meta_skill_context,
        max_completion_tokens,
    )

    if not failure["edits"]:
        return success
    if not success["edits"]:
        return failure

    sources = [failure, success]
    combined = json.dumps(
        [_public_patch(patch) for patch in sources],
        ensure_ascii=False,
        indent=2,
    )
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        "## Two pre-merged patch groups to combine\n"
        f"Group 1 (failure-driven, HIGH priority): "
        f"{len(failure['edits'])} edits\n"
        f"Group 2 (success-driven, lower priority): "
        f"{len(success['edits'])} edits\n\n"
        f"{combined}"
    )
    context = format_meta_skill_context(meta_skill_context)
    if context:
        user = f"{context}\n\n{user}"

    final = _call_merge(
        _system("merge_final"),
        user,
        sources,
        max_completion_tokens,
    )
    if final is not None:
        return final

    return _fallback(
        sources,
        "fallback: failure first, then success",
    )
