"""ReflACT core Reflect engine -- minibatch trajectory analysis.

Provides environment-agnostic minibatch trajectory analysis: instead of
analyzing each trajectory independently, trajectories are grouped into
minibatches of size M and analyzed together -- analogous to minibatch SGD
vs per-sample SGD in neural network training.

Two-level prompt priority system:

1. **Custom prompt** (adapter returns non-None) -- used as-is.
2. **Generic default prompt** (adapter returns None) -- built-in defaults
   that work for any environment without configuration.

Public API
----------
- :func:`fmt_trajectory`               -- format one conversation into text
- :func:`fmt_minibatch_trajectories`   -- format multiple trajectories for batch analysis
- :func:`run_error_analyst_minibatch`   -- one optimizer call for a group of failures
- :func:`run_success_analyst_minibatch` -- one optimizer call for a group of successes
- :func:`run_minibatch_reflect`         -- full reflect stage dispatcher
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.evaluation.compound_edit_screening import (
    SCHEMA_VERSION as COMPOUND_SCREENING_SCHEMA_VERSION,
    screen_reflect_result,
)
from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.structured_skills import (
    PATCH_SCHEMA_VERSION,
    compute_skill_id,
    parse_patch_payload,
    strict_json_loads,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    truncate_payload,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


try:
    from mod_C0_compress import (
        maybe_compress_text,
    )
except ImportError:
    # Keep standalone SkillOpt use backward compatible when C0 is absent.
    def maybe_compress_text(
        *,
        stage,
        text,
        fixed_messages,
    ):
        return text


# ── Trajectory formatting ────────────────────────────────────────────────────

def _trajectory_text(value) -> str:
    """Render optional trajectory fields without discarding evidence."""
    if value is None:
        return ""
    return str(value)


def _clip_text(value, limit: int | None = None) -> str:
    """Compatibility wrapper that intentionally performs no clipping."""
    del limit
    return _trajectory_text(value)


def _format_trajectory_item(
    item,
) -> list[str]:
    """Format one trajectory item without mechanical truncation."""
    if not isinstance(item, dict):
        return [
            f"[agent] {_trajectory_text(item)}",
        ]

    if item.get("type") == "tool_call":
        return [
            f"[action] {_trajectory_text(item.get('cmd'))}",
            f"[obs]    {_trajectory_text(item.get('obs'))}",
        ]

    if (
        "action" in item
        and "env_feedback" in item
    ):
        step = item.get("step", "?")
        reasoning = _trajectory_text(
            item.get("reasoning")
        )
        action = _trajectory_text(
            item.get("action")
        )
        feedback = _trajectory_text(
            item.get("env_feedback")
        )

        lines: list[str] = []
        if reasoning:
            lines.append(
                f"[step {step} think] {reasoning}"
            )
        lines.extend(
            [
                f"[step {step} action] {action}",
                f"[step {step} obs]    {feedback}",
            ]
        )
        return lines

    content = _trajectory_text(
        item.get("content")
    )
    if item.get("role") == "system":
        return [
            f"[verification] {content}",
        ]

    role = item.get("role", "agent")
    return [
        f"[{role}] {content}",
    ]


def fmt_trajectory(
    conversation: list[dict],
    max_chars: int | None = None,
) -> str:
    """Format a complete conversation without mechanical truncation.

    ``max_chars`` is retained only for call compatibility. Evidence selection
    and context compression are handled after formatting.
    """
    del max_chars

    lines: list[str] = []
    for item in conversation:
        lines.extend(
            _format_trajectory_item(item)
        )

    return "\n".join(lines)


def fmt_trajectory_sections(
    conversation: list[dict],
) -> tuple[str, str]:
    """Split a complete trajectory by explicit evidence priority."""
    protected_lines: list[str] = []
    compressible_lines: list[str] = []

    for item in conversation:
        lines = _format_trajectory_item(item)
        is_protected = (
            isinstance(item, dict)
            and item.get("evidence_priority")
            == "protected"
        )
        target = (
            protected_lines
            if is_protected
            else compressible_lines
        )
        target.extend(lines)

    return (
        "\n".join(protected_lines),
        "\n".join(compressible_lines),
    )


def _fmt_forward_math_initial_solver_trajectory(
    item: dict,
    *,
    index: int,
) -> str:
    """
    Format only the evidence needed to improve the initial math solver.

    Forward verification is used only as error evidence. Verification
    flows, execution records, tool logs, and target prompts are excluded.
    """
    task_id = str(
        item.get("id")
        or ""
    )

    problem = str(
        item.get("task_description")
        or item.get("question")
        or item.get("instruction")
        or ""
    ).strip()

    initial_response = str(
        item.get("initial_response")
        or ""
    ).strip()

    initial_answer = str(
        item.get("initial_answer")
        or ""
    ).strip()

    steps = item.get("forward_steps") or []
    if not isinstance(steps, list):
        steps = []

    first_feedback = ""
    revised_response = ""
    revised_answer = ""

    for step in steps:
        if not isinstance(step, dict):
            continue

        feedback = str(
            step.get("F_f")
            or ""
        ).strip()

        if not feedback:
            continue

        first_feedback = feedback
        revised_response = str(
            step.get("y_prime")
            or ""
        ).strip()
        revised_answer = str(
            step.get("a_prime")
            or ""
        ).strip()
        break

    # Defensive fallback for an older or incomplete Forward result.
    if not first_feedback:
        first_feedback = str(
            item.get("fail_reason")
            or ""
        ).strip()

    # In case the first feedback and first repair are stored in
    # different Forward steps, retain the earliest available repair.
    if not revised_response:
        for step in steps:
            if not isinstance(step, dict):
                continue

            candidate = str(
                step.get("y_prime")
                or ""
            ).strip()

            if candidate:
                revised_response = candidate
                revised_answer = str(
                    step.get("a_prime")
                    or ""
                ).strip()
                break

    final_response = str(
        item.get("final_response")
        or ""
    ).strip()

    final_answer = str(
        item.get("final_answer")
        or ""
    ).strip()

    parts = [
        f"### Trajectory {index} (id={task_id})",
        "",
        "[Problem]",
        _clip_text(
            problem,
            4000,
        ),
        "",
        "[Initial Response]",
        _clip_text(
            initial_response,
            6000,
        ),
        "",
        "[Initial Answer]",
        _clip_text(
            initial_answer,
            1000,
        ),
        "",
        "[Forward Pass Signals]",
        f"trajectory_type: {item.get('trajectory_type', '')}",
        f"hard: {item.get('hard', '')}",
        (
            "forward_initial_pass: "
            f"{item.get('forward_initial_pass', '')}"
        ),
        (
            "forward_final_pass: "
            f"{item.get('forward_final_pass', '')}"
        ),
        (
            "forward_repair_attempted: "
            f"{item.get('forward_repair_attempted', '')}"
        ),
        (
            "forward_repair_accepted: "
            f"{item.get('forward_repair_accepted', '')}"
        ),
        (
            "verification_unavailable_reason: "
            f"{item.get('verification_unavailable_reason', '')}"
        ),
    ]

    if first_feedback:
        parts.extend(
            [
                "",
                "[First Forward Feedback]",
                _clip_text(
                    first_feedback,
                    3000,
                ),
            ]
        )

    if revised_response:
        parts.extend(
            [
                "",
                "[Revised Response]",
                _clip_text(
                    revised_response,
                    6000,
                ),
            ]
        )

    if revised_answer:
        parts.extend(
            [
                "",
                "[Revised Answer]",
                _clip_text(
                    revised_answer,
                    1000,
                ),
            ]
        )

    # Avoid repeating an identical initial or revised response.
    if (
        final_response
        and final_response != initial_response
        and final_response != revised_response
    ):
        parts.extend(
            [
                "",
                "[Final Response]",
                _clip_text(
                    final_response,
                    6000,
                ),
            ]
        )

    if (
        final_answer
        and final_answer != initial_answer
        and final_answer != revised_answer
    ):
        parts.extend(
            [
                "",
                "[Final Answer]",
                _clip_text(
                    final_answer,
                    1000,
                ),
            ]
        )

    return "\n".join(parts)


def _fmt_forward_math_initial_solver_trajectory_outputs(
    item: dict,
    *,
    index: int,
) -> tuple[str, str, str]:
    """Build S5 initial-solver full, protected, and supporting views.

    The Forward workflow remains error evidence only. Internal verification
    flows, execution records, tool logs, and S8 streaming-teacher fields are
    deliberately excluded from every analyst view while remaining available
    in the raw rollout ledger on disk.
    """
    full_text = _fmt_forward_math_initial_solver_trajectory(
        item,
        index=index,
    )

    task_id = str(item.get("id") or "")
    problem = str(
        item.get("task_description")
        or item.get("question")
        or item.get("instruction")
        or ""
    ).strip()
    initial_response = str(
        item.get("initial_response") or ""
    ).strip()
    initial_answer = str(
        item.get("initial_answer") or ""
    ).strip()

    steps = item.get("forward_steps") or []
    if not isinstance(steps, list):
        steps = []

    first_feedback = ""
    revised_response = ""
    revised_answer = ""

    for step in steps:
        if not isinstance(step, dict):
            continue
        feedback = str(step.get("F_f") or "").strip()
        if not feedback:
            continue
        first_feedback = feedback
        revised_response = str(
            step.get("y_prime") or ""
        ).strip()
        revised_answer = str(
            step.get("a_prime") or ""
        ).strip()
        break

    if not first_feedback:
        first_feedback = str(
            item.get("fail_reason") or ""
        ).strip()

    if not revised_response:
        for step in steps:
            if not isinstance(step, dict):
                continue
            candidate = str(
                step.get("y_prime") or ""
            ).strip()
            if not candidate:
                continue
            revised_response = candidate
            revised_answer = str(
                step.get("a_prime") or ""
            ).strip()
            break

    final_response = str(
        item.get("final_response") or ""
    ).strip()
    final_answer = str(
        item.get("final_answer") or ""
    ).strip()

    protected_parts = [
        f"### Trajectory {index} (id={task_id})",
        "",
        "[Problem]",
        problem,
        "",
        "[Initial Answer]",
        initial_answer,
        "",
        "[Forward Pass Signals]",
        f"trajectory_type: {item.get('trajectory_type', '')}",
        f"hard: {item.get('hard', '')}",
        (
            "forward_initial_pass: "
            f"{item.get('forward_initial_pass', '')}"
        ),
        (
            "forward_final_pass: "
            f"{item.get('forward_final_pass', '')}"
        ),
        (
            "forward_repair_attempted: "
            f"{item.get('forward_repair_attempted', '')}"
        ),
        (
            "forward_repair_accepted: "
            f"{item.get('forward_repair_accepted', '')}"
        ),
        (
            "verification_unavailable_reason: "
            f"{item.get('verification_unavailable_reason', '')}"
        ),
    ]

    supporting_parts = [
        f"### Trajectory {index} (id={task_id})",
    ]
    if initial_response:
        supporting_parts.extend(
            [
                "",
                "[Initial Response]",
                initial_response,
            ]
        )

    if first_feedback:
        protected_parts.extend(
            [
                "",
                "[First Forward Feedback]",
                first_feedback,
            ]
        )

    if revised_response:
        supporting_parts.extend(
            [
                "",
                "[Revised Response]",
                revised_response,
            ]
        )

    if revised_answer:
        protected_parts.extend(
            [
                "",
                "[Revised Answer]",
                revised_answer,
            ]
        )

    if (
        final_response
        and final_response != initial_response
        and final_response != revised_response
    ):
        supporting_parts.extend(
            [
                "",
                "[Final Response]",
                final_response,
            ]
        )

    if (
        final_answer
        and final_answer != initial_answer
        and final_answer != revised_answer
    ):
        protected_parts.extend(
            [
                "",
                "[Final Answer]",
                final_answer,
            ]
        )

    return (
        full_text,
        "\n".join(protected_parts),
        "\n".join(supporting_parts),
    )


# ── Minibatch trajectory formatting ──────────────────────────────────────────


def _fmt_minibatch_trajectories_legacy(
    items: list[dict],
    prediction_dir: str,
) -> str:
    """Format multiple trajectories for minibatch analyst consumption.

    Each item is a rollout result dict with ``"id"``, ``"task_description"``,
    ``"task_type"``, ``"fail_reason"``, etc.

    Forward-math items are formatted from structured rollout evidence so that
    the analysts see only the problem, initial response, first Forward feedback,
    revised response, and pass signals. Other environments retain the original
    ``conversation.json`` formatting behavior.

    Parameters
    ----------
    items : list[dict]
        Rollout result dicts belonging to one minibatch.
    prediction_dir : str
        Path to ``predictions/`` directory containing per-task
        ``<task_id>/conversation.json`` files.

    Returns
    -------
    str
        Formatted text with all trajectories separated by ``---``.
    """
    parts: list[str] = []
    for idx, item in enumerate(items, 1):
        tid = str(item["id"])
        prediction_id = str(item.get("prediction_id") or tid)

        # S5-7-2: Skill is learned only for the initial math solver.
        # Do not expose complete Forward/tool traces to the analysts.
        if str(
            item.get("task_type")
            or ""
        ).strip() == "forward_math":
            parts.append(
                _fmt_forward_math_initial_solver_trajectory(
                    item,
                    index=idx,
                )
            )
            continue

        conv_path = os.path.join(
            prediction_dir,
            prediction_id,
            "conversation.json",
        )
        if not os.path.exists(conv_path):
            continue
        with open(conv_path, encoding="utf-8") as f:
            conversation = json.load(f)
        if not conversation:
            continue

        traj_text = fmt_trajectory(conversation)
        header = (
            f"### Trajectory {idx} (id={tid})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        reference_text = str(item.get("reference_text") or "").strip()
        if reference_text:
            header += (
                f"\n#### Hidden Reference\n"
                f"{reference_text}\n"
            )

        # ── Append target context (what the agent saw) ──────────────
        target_prompt = item.get("target_system_prompt", "")
        if not target_prompt:
            prompt_path = os.path.join(prediction_dir, prediction_id, "target_system_prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, encoding="utf-8") as f:
                    target_prompt = f.read()
        if target_prompt:
            header += (
                f"\n#### Target System Prompt\n"
                f"{target_prompt}\n"
            )

        user_prompt = item.get("target_user_prompt", "")
        if not user_prompt:
            user_prompt_path = os.path.join(prediction_dir, prediction_id, "target_user_prompt.txt")
            if os.path.exists(user_prompt_path):
                with open(user_prompt_path, encoding="utf-8") as f:
                    user_prompt = f.read()
        if user_prompt:
            header += (
                f"\n#### Target User Prompt\n"
                f"{user_prompt}\n"
            )

        if os.environ.get("REFLACT_CODEX_TRACE_TO_OPTIMIZER", "0") == "1":
            codex_trace_summary = item.get("codex_trace_summary", "")
            if not codex_trace_summary:
                codex_trace_summary_path = os.path.join(prediction_dir, prediction_id, "codex_trace_summary.txt")
                if os.path.exists(codex_trace_summary_path):
                    with open(codex_trace_summary_path, encoding="utf-8") as f:
                        codex_trace_summary = f.read()
            if codex_trace_summary:
                header += (
                    f"\n#### Codex Trace Summary\n"
                    f"{codex_trace_summary}\n"
                )

        codex_probe_trace_steps = str(item.get("codex_probe_trace_steps") or "").strip()
        if codex_probe_trace_steps:
            header += (
                f"\n#### Codex Trace Steps\n"
                f"{codex_probe_trace_steps}\n"
            )

        preview = item.get("spreadsheet_preview", "")
        if not preview:
            preview_path = os.path.join(prediction_dir, prediction_id, "spreadsheet_preview.txt")
            if os.path.exists(preview_path):
                with open(preview_path, encoding="utf-8") as f:
                    preview = f.read()
        if preview:
            header += (
                f"\n#### Spreadsheet Preview\n"
                f"{preview}\n"
            )

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


def _fmt_minibatch_trajectory_outputs(
    items: list[dict],
    prediction_dir: str,
) -> tuple[str, str, str]:
    """Build full, protected, and compressible minibatch views."""
    full_parts: list[str] = []
    protected_parts: list[str] = []
    compressible_parts: list[str] = []

    for index, item in enumerate(items, 1):
        task_id = str(item["id"])
        prediction_id = str(
            item.get("prediction_id") or task_id
        )

        if str(
            item.get("task_type") or ""
        ).strip() == "forward_math":
            full, protected, supporting = (
                _fmt_forward_math_initial_solver_trajectory_outputs(
                    item,
                    index=index,
                )
            )
            full_parts.append(full)
            protected_parts.append(protected)
            if supporting.strip():
                compressible_parts.append(supporting)
            continue

        conversation_path = os.path.join(
            prediction_dir,
            prediction_id,
            "conversation.json",
        )
        if not os.path.exists(conversation_path):
            continue
        with open(
            conversation_path,
            encoding="utf-8",
        ) as file:
            conversation = json.load(file)
        if not conversation:
            continue

        protected_header = (
            f"### Trajectory {index} (id={task_id})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = _trajectory_text(
            item.get("fail_reason", "")
        )
        if fail_reason:
            protected_header += (
                f"Failure reason: {fail_reason}\n"
            )
        protected_header += (
            f"Steps: {item.get('n_turns', '?')}\n"
        )

        reference_text = _trajectory_text(
            item.get("reference_text")
        )
        if reference_text.strip():
            protected_header += (
                "\n#### Hidden Reference\n"
                f"{reference_text}\n"
            )

        supporting_header_parts: list[str] = []

        target_prompt = item.get(
            "target_system_prompt",
            "",
        )
        if not target_prompt:
            target_prompt_path = os.path.join(
                prediction_dir,
                prediction_id,
                "target_system_prompt.txt",
            )
            if os.path.exists(target_prompt_path):
                with open(
                    target_prompt_path,
                    encoding="utf-8",
                ) as file:
                    target_prompt = file.read()
        target_prompt_text = _trajectory_text(
            target_prompt
        )
        if target_prompt_text.strip():
            supporting_header_parts.append(
                "#### Target System Prompt\n"
                f"{target_prompt_text}"
            )

        user_prompt = item.get(
            "target_user_prompt",
            "",
        )
        if not user_prompt:
            user_prompt_path = os.path.join(
                prediction_dir,
                prediction_id,
                "target_user_prompt.txt",
            )
            if os.path.exists(user_prompt_path):
                with open(
                    user_prompt_path,
                    encoding="utf-8",
                ) as file:
                    user_prompt = file.read()
        user_prompt_text = _trajectory_text(
            user_prompt
        )
        if user_prompt_text.strip():
            supporting_header_parts.append(
                "#### Target User Prompt\n"
                f"{user_prompt_text}"
            )

        if os.environ.get(
            "REFLACT_CODEX_TRACE_TO_OPTIMIZER",
            "0",
        ) == "1":
            codex_trace_summary = item.get(
                "codex_trace_summary",
                "",
            )
            if not codex_trace_summary:
                trace_summary_path = os.path.join(
                    prediction_dir,
                    prediction_id,
                    "codex_trace_summary.txt",
                )
                if os.path.exists(trace_summary_path):
                    with open(
                        trace_summary_path,
                        encoding="utf-8",
                    ) as file:
                        codex_trace_summary = file.read()
            codex_trace_summary_text = _trajectory_text(
                codex_trace_summary
            )
            if codex_trace_summary_text.strip():
                supporting_header_parts.append(
                    "#### Codex Trace Summary\n"
                    f"{codex_trace_summary_text}"
                )

        codex_trace_steps = _trajectory_text(
            item.get("codex_probe_trace_steps")
        )
        if codex_trace_steps.strip():
            supporting_header_parts.append(
                "#### Codex Trace Steps\n"
                f"{codex_trace_steps}"
            )

        preview = item.get(
            "spreadsheet_preview",
            "",
        )
        if not preview:
            preview_path = os.path.join(
                prediction_dir,
                prediction_id,
                "spreadsheet_preview.txt",
            )
            if os.path.exists(preview_path):
                with open(
                    preview_path,
                    encoding="utf-8",
                ) as file:
                    preview = file.read()
        preview_text = _trajectory_text(preview)
        if preview_text.strip():
            supporting_header_parts.append(
                "#### Spreadsheet Preview\n"
                f"{preview_text}"
            )

        full_text = fmt_trajectory(conversation)
        protected_text, compressible_text = (
            fmt_trajectory_sections(conversation)
        )

        full_part = protected_header
        if supporting_header_parts:
            full_part += (
                "\n"
                + "\n\n".join(
                    supporting_header_parts
                )
                + "\n"
            )
        if full_text:
            full_part += "\n" + full_text
        full_parts.append(full_part)

        protected_part = protected_header
        if protected_text:
            protected_part += "\n" + protected_text
        protected_parts.append(protected_part)

        supporting_sections = list(
            supporting_header_parts
        )
        if compressible_text:
            supporting_sections.append(
                compressible_text
            )
        if supporting_sections:
            compressible_parts.append(
                f"### Trajectory {index} (id={task_id})\n"
                + "\n\n".join(supporting_sections)
            )

    separator = "\n\n---\n\n"
    return (
        separator.join(full_parts),
        separator.join(protected_parts),
        separator.join(compressible_parts),
    )


def fmt_minibatch_trajectories(
    items: list[dict],
    prediction_dir: str,
) -> str:
    """Return the complete untruncated minibatch view."""
    full_text, _, _ = _fmt_minibatch_trajectory_outputs(
        items,
        prediction_dir,
    )
    return full_text


def fmt_minibatch_trajectory_sections(
    items: list[dict],
    prediction_dir: str,
) -> tuple[str, str]:
    """Return protected evidence and compressible supporting details."""
    _, protected, supporting = (
        _fmt_minibatch_trajectory_outputs(
            items,
            prediction_dir,
        )
    )
    return protected, supporting


# ── Prompt resolution ───────────────────────────────────────────────────────


def _resolve_prompt(custom: str | None, default_name: str, update_mode: str = "patch") -> str:
    """Return *custom* if provided (non-None), otherwise load from file."""
    if custom is not None:
        return custom
    mode = normalize_update_mode(update_mode)
    actual_name = default_name
    if is_full_rewrite_minibatch_mode(mode):
        full_name = f"{default_name}_full_rewrite"
        try:
            return load_prompt(full_name)
        except FileNotFoundError:
            actual_name = default_name
    elif mode == "rewrite_from_suggestions":
        rewrite_name = f"{default_name}_rewrite"
        try:
            return load_prompt(rewrite_name)
        except FileNotFoundError:
            actual_name = default_name
    return load_prompt(actual_name)


# ── Minibatch analysts ──────────────────────────────────────────────────────


_ANALYST_OUTPUT_FORMAT_REMINDER = (
    "## Output Format Reminder\n\n"
    "Use the complete original evidence and instructions above to generate "
    "the result. The task and evidence requirements remain unchanged.\n\n"
    'Return only one valid JSON object with a top-level "patch" field. '
    f'The patch.schema_version must be "{PATCH_SCHEMA_VERSION}". '
    'Every non-delete Edit must contain a non-empty "skills" array of '
    'strict two-field JSON skill objects. '
    "Do not include Markdown code fences or any text outside the JSON object."
)

_ANALYST_STRICT_OUTPUT_FORMAT_REMINDER = (
    "## Strict Output Format Reminder\n\n"
    "Analyze the complete original evidence and instructions above and "
    "strictly follow the specified JSON format.\n\n"
    'The output must be a JSON object with a top-level "patch" field, '
    'and "patch" must exactly follow structured patch schema version '
    f'"{PATCH_SCHEMA_VERSION}". Do not output Markdown skill text, evidence, '
    'provenance, support counts, or semantic hashes.\n\n'
    "Do not include Markdown code fences, explanations, or any text outside "
    "the JSON object."
)


def _build_analyst_user_prompt(
    original_user: str,
    round_index: int,
) -> str:
    """Build cumulative Analyst prompts: A, A+R1, A+R1+R2."""
    if round_index <= 0:
        return original_user

    retry_user = (
        f"{original_user}\n\n"
        f"{_ANALYST_OUTPUT_FORMAT_REMINDER}"
    )

    if round_index >= 2:
        retry_user += (
            f"\n\n"
            f"{_ANALYST_STRICT_OUTPUT_FORMAT_REMINDER}"
        )

    return retry_user


def _run_analyst_with_format_retry(
    *,
    system: str,
    user: str,
    max_completion_tokens: int,
    max_analyst_rounds: int,
    update_mode: str,
    known_skill_ids: set[str] | None = None,
) -> dict | None:
    """Retry malformed Analyst output with cumulative format reminders."""
    total_rounds = min(
        3,
        max(
            1,
            int(max_analyst_rounds or 3),
        ),
    )

    for round_index in range(total_rounds):
        response, _ = chat_optimizer(
            system=system,
            user=_build_analyst_user_prompt(
                user,
                round_index,
            ),
            max_completion_tokens=max_completion_tokens,
            retries=4,
            stage="analyst",
        )

        if normalize_update_mode(update_mode) == "patch":
            try:
                result = strict_json_loads(response)
                if not isinstance(result, dict):
                    raise TypeError(
                        "Analyst output must be a JSON object."
                    )
                patch = parse_patch_payload(
                    result.get("patch"),
                    known_skill_ids=known_skill_ids,
                    allow_ranking_details=False,
                    allow_trusted_metadata=False,
                )
                if any(
                    edit["op"] == "replace"
                    and len(edit.get("skills", [])) != 1
                    for edit in patch["edits"]
                ):
                    raise ValueError(
                        "A replace Edit must contain exactly one skill."
                    )
                result = dict(result)
                result["patch"] = patch
            except (TypeError, ValueError):
                result = None
        else:
            result = extract_json(response)

        if (
            isinstance(result, dict)
            and isinstance(result.get("patch"), dict)
        ):
            return result

        if round_index + 1 < total_rounds:
            print(
                "      [analyst] invalid JSON/patch format; "
                "retrying with stronger output-format instructions"
            )

    return None


def _known_skill_ids_from_runtime_skill(
    skill_content: str,
) -> set[str] | None:
    """Read structured skill IDs without a Markdown fallback."""
    try:
        payload = strict_json_loads(skill_content)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("skills"), list
        ):
            return None
        return {
            str(skill["skill_id"])
            for skill in payload["skills"]
            if isinstance(skill, dict)
            and isinstance(skill.get("skill_id"), str)
        }
    except (KeyError, TypeError, ValueError):
        return None


def _verified_evidence_sidecar(
    patch: dict,
    *,
    items: list[dict],
    source_type: str,
) -> dict[str, list[dict[str, str]]]:
    """Bind trusted minibatch evidence to semantic skill IDs in Python."""
    evidence: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        evidence_id = str(item["id"]).strip()
        if not evidence_id or evidence_id in seen_ids:
            continue
        seen_ids.add(evidence_id)
        evidence.append({
            "evidence_id": evidence_id,
            "source_type": source_type,
        })

    sidecar: dict[str, list[dict[str, str]]] = {}
    for edit in patch.get("edits", []):
        for skill in edit.get("skills", []):
            sidecar[compute_skill_id(skill)] = copy.deepcopy(evidence)
    return sidecar


def _attach_verified_evidence(
    patch: dict,
    *,
    items: list[dict],
    source_type: str,
) -> dict:
    """Attach only program-derived evidence after model validation."""
    canonical = parse_patch_payload(
        patch,
        allow_ranking_details=True,
        allow_trusted_metadata=False,
    )
    canonical["_evidence_sidecar"] = _verified_evidence_sidecar(
        canonical,
        items=items,
        source_type=source_type,
    )
    return parse_patch_payload(
        canonical,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def _reconcile_filtered_evidence(
    filtered_patch: dict,
    source_patch: dict,
    screening_result: dict | None = None,
) -> dict:
    """Re-key trusted evidence after a program-mediated role repair."""
    if (
        not isinstance(filtered_patch, dict)
        or not isinstance(source_patch, dict)
        or filtered_patch.get("schema_version") != PATCH_SCHEMA_VERSION
        or source_patch.get("schema_version") != PATCH_SCHEMA_VERSION
    ):
        return filtered_patch

    source = parse_patch_payload(
        source_patch,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )
    public_filtered = {
        key: copy.deepcopy(value)
        for key, value in filtered_patch.items()
        if key != "_evidence_sidecar"
    }
    filtered = parse_patch_payload(
        public_filtered,
        allow_ranking_details=True,
        allow_trusted_metadata=False,
    )

    source_sidecar = source.get("_evidence_sidecar", {})
    source_id_by_output_id: dict[str, str] = {}
    if isinstance(screening_result, dict):
        for parent_result in screening_result.get("results", []):
            if not isinstance(parent_result, dict):
                continue
            role = parent_result.get("role")
            if not isinstance(role, dict):
                continue
            for role_result in role.get("results", []):
                if not isinstance(role_result, dict):
                    continue
                source_skill_id = str(
                    role_result.get("source_skill_id") or ""
                )
                output_skill_id = str(
                    role_result.get("output_skill_id") or ""
                )
                if source_skill_id and output_skill_id:
                    source_id_by_output_id[output_skill_id] = source_skill_id

    sidecar: dict[str, list[dict[str, str]]] = {}
    for edit in filtered["edits"]:
        for skill in edit.get("skills", []):
            skill_id = compute_skill_id(skill)
            source_skill_id = source_id_by_output_id.get(skill_id, skill_id)
            if source_skill_id not in source_sidecar:
                raise ValueError(
                    "Cannot bind trusted evidence to a screened atomic "
                    f"skill: output={skill_id!r}, source={source_skill_id!r}."
                )
            sidecar[skill_id] = copy.deepcopy(
                source_sidecar[source_skill_id]
            )
    filtered["_evidence_sidecar"] = sidecar
    return parse_patch_payload(
        filtered,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )


def _append_selectively_compressed_trajectories(
    *,
    user: str,
    actual_system: str,
    protected_evidence: str,
    compressible_details: str,
    stage: str,
    step_buffer_context: str = "",
) -> str:
    """Keep protected evidence verbatim and compress supporting details."""
    user_with_protected = (
        f"{user}"
        "### Protected Evidence\n"
        f"{protected_evidence}\n\n"
        "### Supporting Details\n"
    )

    supporting_details = compressible_details
    if step_buffer_context.strip():
        previous_steps = (
            "#### Previous Steps in This Epoch\n"
            f"{step_buffer_context}"
        )
        supporting_details = (
            previous_steps
            + (
                "\n\n" + compressible_details
                if compressible_details.strip()
                else ""
            )
        )

    if not supporting_details.strip():
        return user_with_protected

    compressed_details = maybe_compress_text(
        stage=stage,
        text=supporting_details,
        fixed_messages=[
            {
                "role": "system",
                "content": actual_system,
            },
            {
                "role": "user",
                "content": user_with_protected,
            },
        ],
    )

    return user_with_protected + compressed_details


def run_error_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    max_completion_tokens: int = 0,
    max_analyst_rounds: int = 3,
) -> dict | None:
    """Analyze a minibatch of failed trajectories in one optimizer call.

    Parameters
    ----------
    skill_content : str
        Current skill document text.
    items : list[dict]
        Rollout result dicts (all should have ``hard=0``).
    prediction_dir : str
        Path to ``predictions/`` directory.
    edit_budget : int
        Maximum number of edits (L).
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    rejection_context : str
        *Deprecated* — use ``step_buffer_context``.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="failure"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_error", mode)

    protected_evidence, supporting_details = (
        fmt_minibatch_trajectory_sections(
            items,
            prediction_dir,
        )
    )
    if not (
        protected_evidence.strip()
        or supporting_details.strip()
    ):
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    # Unified step buffer context (preferred)
    ctx = step_buffer_context or rejection_context or ""
    if trajectory_memory_context:
        ctx = f"{ctx}\n{trajectory_memory_context}" if ctx else trajectory_memory_context
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    user += (
        f"## Failed Trajectories ({len(items)} total)\n\n"
    )

    user = _append_selectively_compressed_trajectories(
        user=user,
        actual_system=actual_system,
        protected_evidence=protected_evidence,
        compressible_details=supporting_details,
        stage="SkillOpt.reflect.failure",
        step_buffer_context=ctx,
    )

    try:
        result = _run_analyst_with_format_retry(
            system=actual_system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            max_analyst_rounds=max_analyst_rounds,
            update_mode=mode,
            known_skill_ids=_known_skill_ids_from_runtime_skill(
                skill_content
            ),
        )

        if result is not None:
            result["source_type"] = "failure"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(
                    result["patch"],
                    edit_budget,
                    mode,
                )
            if mode == "patch":
                result["patch"] = _attach_verified_evidence(
                    result["patch"],
                    items=items,
                    source_type="failure",
                )
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None


def run_success_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    max_completion_tokens: int = 0,
    max_analyst_rounds: int = 3,
) -> dict | None:
    """Analyze a minibatch of successful trajectories in one optimizer call.

    Parameters
    ----------
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="success"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_success", mode)

    protected_evidence, supporting_details = (
        fmt_minibatch_trajectory_sections(
            items,
            prediction_dir,
        )
    )
    if not (
        protected_evidence.strip()
        or supporting_details.strip()
    ):
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    ctx = step_buffer_context or trajectory_memory_context or ""
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    user += (
        f"## Successful Trajectories ({len(items)} total)\n\n"
    )

    user = _append_selectively_compressed_trajectories(
        user=user,
        actual_system=actual_system,
        protected_evidence=protected_evidence,
        compressible_details=supporting_details,
        stage="SkillOpt.reflect.success",
        step_buffer_context=ctx,
    )

    try:
        result = _run_analyst_with_format_retry(
            system=actual_system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            max_analyst_rounds=max_analyst_rounds,
            update_mode=mode,
            known_skill_ids=_known_skill_ids_from_runtime_skill(
                skill_content
            ),
        )

        if result is not None:
            result["source_type"] = "success"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(
                    result["patch"],
                    edit_budget,
                    mode,
                )
            if mode == "patch":
                result["patch"] = _attach_verified_evidence(
                    result["patch"],
                    items=items,
                    source_type="success",
                )
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None


# ── Minibatch reflect dispatcher ────────────────────────────────────────────


def _split_minibatches(items: list, batch_size: int) -> list[list]:
    """Split items into minibatches of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _shuffle_for_minibatch(items: list, seed: int | None) -> list:
    """Return items in minibatch order.

    Uses a deterministic shuffle when a seed is provided so resume runs keep
    the same minibatch composition. Falls back to input order when no seed is
    available.
    """
    ordered = list(items)
    if seed is None:
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered


_COMPOUND_SCREENING_PROMPT_NAMES = (
    "role_consistency_role",
    "role_consistency_check",
    "role_consistency_repair",
    "role_consistency_postcheck",
)


class CompoundScreeningTechnicalError(
    RuntimeError
):
    """Pre-Aggregate screening failed without a semantic decision."""


def _canonical_json(value) -> str:
    """Serialize cache-key material deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _compound_screening_policy_hash(
    env: str,
) -> str:
    policy_parts = [
        COMPOUND_SCREENING_SCHEMA_VERSION,
    ]
    policy_parts.extend(
        load_prompt(name, env=env)
        for name in _COMPOUND_SCREENING_PROMPT_NAMES
    )
    return _sha256_text(
        "\0".join(policy_parts)
    )


def _compound_screening_input_hash(
    raw_result,
    *,
    exact_minibatch_rows: list[dict],
    policy_hash: str,
    env: str,
    format_max_attempts: int,
    max_completion_tokens: int,
) -> str:
    evidence_rows_hash = _sha256_text(
        _canonical_json(exact_minibatch_rows)
    )
    payload = {
        "schema_version": COMPOUND_SCREENING_SCHEMA_VERSION,
        "policy_hash": policy_hash,
        "patch": raw_result,
        "evidence_rows_hash": evidence_rows_hash,
        "env": env,
        "format_max_attempts": format_max_attempts,
        "max_completion_tokens": max_completion_tokens,
    }
    return _sha256_text(
        _canonical_json(payload)
    )


def _load_compound_screening_cache(
    *,
    filtered_path: str,
    audit_path: str,
    input_hash: str,
) -> dict | None:
    if not (
        os.path.isfile(filtered_path)
        and os.path.isfile(audit_path)
    ):
        return None

    try:
        with open(
            audit_path,
            encoding="utf-8",
        ) as file:
            audit = json.load(file)
        with open(
            filtered_path,
            encoding="utf-8",
        ) as file:
            filtered_result = json.load(
                file
            )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(audit, dict):
        return None
    if audit.get("input_hash") != input_hash:
        return None
    if audit.get("cacheable") is not True:
        return None

    audit_result = audit.get("result")
    if not isinstance(audit_result, dict):
        return None
    if audit_result.get("passed") is not True:
        return None

    if not isinstance(filtered_result, dict):
        return None
    if not isinstance(
        filtered_result.get("patch"),
        dict,
    ):
        return None

    marker = filtered_result.get(
        "_preaggregate_compound_screening"
    )
    if not isinstance(marker, dict):
        return None
    if marker.get("schema_version") != (
        COMPOUND_SCREENING_SCHEMA_VERSION
    ):
        return None
    if marker.get("passed") is not True:
        return None

    return filtered_result


def _compound_screening_exception_result(
    raw_result,
    *,
    error: str,
) -> dict:
    filtered_result = (
        copy.deepcopy(raw_result)
        if isinstance(raw_result, dict)
        else {"patch": {"edits": []}}
    )
    inner_patch = filtered_result.get("patch")
    if not isinstance(inner_patch, dict):
        inner_patch = {}
        filtered_result["patch"] = inner_patch
    inner_patch["edits"] = []

    summary = {
        "input_parent_edit_count": 0,
        "output_parent_edit_count": 0,
        "technical_parent_drop_count": 0,
        "semantic_parent_drop_count": 0,
        "delete_passthrough_count": 0,
        "cacheable": False,
    }
    filtered_result[
        "_preaggregate_compound_screening"
    ] = {
        "schema_version": COMPOUND_SCREENING_SCHEMA_VERSION,
        "passed": False,
        "status": "screening_exception",
        "summary": summary,
        "error": error,
    }
    return {
        "passed": False,
        "status": "screening_exception",
        "changed": True,
        "cacheable": False,
        "filtered_result": filtered_result,
        "summary": summary,
        "results": [],
        "error": error,
    }


def _write_compound_screening_sidecars(
    *,
    filtered_path: str,
    audit_path: str,
    input_hash: str,
    screening_result: dict,
) -> None:
    audit_result = {
        key: copy.deepcopy(value)
        for key, value in screening_result.items()
        if key != "filtered_result"
    }
    cacheable = (
        screening_result.get("passed") is True
        and screening_result.get("cacheable") is True
    )
    audit = {
        "input_hash": input_hash,
        "cacheable": cacheable,
        "result": audit_result,
    }

    if cacheable:
        filtered_result = screening_result[
            "filtered_result"
        ]
        with open(
            filtered_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                filtered_result,
                file,
                ensure_ascii=False,
                indent=2,
            )
    else:
        # A failed screening result must never coexist with a stale
        # cacheable filtered payload.
        try:
            os.remove(filtered_path)
        except FileNotFoundError:
            pass

    with open(
        audit_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _run_preaggregate_compound_screening(
    raw_patch_entries: list[dict],
    *,
    patches_dir: str,
    workers: int,
    env: str,
    format_max_attempts: int,
    max_completion_tokens: int,
) -> list[dict]:
    screening_dir = os.path.join(
        patches_dir,
        "preaggregate_compound_screening",
    )
    os.makedirs(screening_dir, exist_ok=True)

    policy_hash = ""
    policy_error = ""
    try:
        policy_hash = (
            _compound_screening_policy_hash(env)
        )
    except Exception as exc:  # noqa: BLE001
        policy_error = (
            "Unable to load compound-screening policy: "
            f"{type(exc).__name__}: {exc}"
        )

    filtered_results: list[dict | None] = [
        None
        for _ in raw_patch_entries
    ]
    pending: list[
        tuple[int, dict, str, str, str]
    ] = []

    for entry_index, entry in enumerate(
        raw_patch_entries
    ):
        tag = str(entry["tag"])
        raw_result = entry["raw_result"]
        exact_rows = entry["exact_minibatch_rows"]
        filtered_path = os.path.join(
            screening_dir,
            f"{tag}.filtered.json",
        )
        audit_path = os.path.join(
            screening_dir,
            f"{tag}.audit.json",
        )

        input_hash = ""
        input_hash_error = ""
        try:
            input_hash = (
                _compound_screening_input_hash(
                    raw_result,
                    exact_minibatch_rows=exact_rows,
                    policy_hash=policy_hash,
                    env=env,
                    format_max_attempts=(
                        format_max_attempts
                    ),
                    max_completion_tokens=(
                        max_completion_tokens
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            input_hash_error = (
                "Unable to build compound-screening input hash: "
                f"{type(exc).__name__}: {exc}"
            )

        technical_error = (
            policy_error
            or input_hash_error
        )
        if technical_error:
            screening_result = (
                _compound_screening_exception_result(
                    raw_result,
                    error=technical_error,
                )
            )
            _write_compound_screening_sidecars(
                filtered_path=filtered_path,
                audit_path=audit_path,
                input_hash=input_hash,
                screening_result=screening_result,
            )
            filtered_results[entry_index] = (
                screening_result["filtered_result"]
            )
            print(
                f"      [compound screening] {tag} "
                "→ screening_exception (fail-closed)"
            )
            continue

        cached_result = (
            _load_compound_screening_cache(
                filtered_path=filtered_path,
                audit_path=audit_path,
                input_hash=input_hash,
            )
        )
        if cached_result is not None:
            filtered_results[entry_index] = (
                cached_result
            )
            marker = cached_result.get(
                "_preaggregate_compound_screening",
                {},
            )
            print(
                f"      [compound screening] {tag} "
                f"→ {marker.get('status', 'cached')} "
                "(resumed)"
            )
            continue

        pending.append(
            (
                entry_index,
                entry,
                input_hash,
                filtered_path,
                audit_path,
            )
        )

    def _screen_one(
        entry: dict,
    ) -> dict:
        try:
            screening_result = screen_reflect_result(
                copy.deepcopy(entry["raw_result"]),
                exact_minibatch_rows=copy.deepcopy(
                    entry["exact_minibatch_rows"]
                ),
                batch_tag=str(entry["tag"]),
                env=env,
                max_completion_tokens=(
                    max_completion_tokens
                ),
                format_max_attempts=(
                    format_max_attempts
                ),
            )
            if not isinstance(screening_result, dict):
                raise TypeError(
                    "Compound screening must return a dictionary."
                )
            filtered_result = screening_result.get(
                "filtered_result"
            )
            if not isinstance(filtered_result, dict):
                raise TypeError(
                    "Compound screening returned no filtered_result dictionary."
                )
            if not isinstance(
                filtered_result.get("patch"),
                dict,
            ):
                raise TypeError(
                    "Compound screening returned an invalid filtered_result.patch."
                )
            filtered_result["patch"] = _reconcile_filtered_evidence(
                filtered_result["patch"],
                entry["raw_result"]["patch"],
                screening_result,
            )
            return screening_result
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return _compound_screening_exception_result(
                entry["raw_result"],
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    if pending:
        with ThreadPoolExecutor(
            max_workers=workers
        ) as ex:
            futs = {
                ex.submit(
                    _screen_one,
                    entry,
                ): (
                    entry_index,
                    entry,
                    input_hash,
                    filtered_path,
                    audit_path,
                )
                for (
                    entry_index,
                    entry,
                    input_hash,
                    filtered_path,
                    audit_path,
                ) in pending
            }

            for completed_index, fut in enumerate(
                as_completed(futs),
                1,
            ):
                (
                    entry_index,
                    entry,
                    input_hash,
                    filtered_path,
                    audit_path,
                ) = futs[fut]
                screening_result = fut.result()
                _write_compound_screening_sidecars(
                    filtered_path=filtered_path,
                    audit_path=audit_path,
                    input_hash=input_hash,
                    screening_result=screening_result,
                )
                filtered_results[entry_index] = (
                    screening_result["filtered_result"]
                )
                summary = screening_result.get(
                    "summary",
                    {},
                )
                print(
                    "      [compound screening] "
                    f"{completed_index}/{len(pending)} "
                    f"{entry['tag']} → "
                    f"{screening_result.get('status', 'unknown')} "
                    "(parents "
                    f"{summary.get('input_parent_edit_count', 0)}"
                    "→"
                    f"{summary.get('output_parent_edit_count', 0)})"
                )

    if any(
        result is None
        for result in filtered_results
    ):
        raise CompoundScreeningTechnicalError(
            "Pre-Aggregate compound screening produced an "
            "incomplete internal result. The current step was "
            "not committed."
        )

    ordered_results = [
        result
        for result in filtered_results
        if result is not None
    ]

    technical_failures: list[str] = []

    for result in ordered_results:
        marker = result.get(
            "_preaggregate_compound_screening"
        )

        if not isinstance(marker, dict):
            technical_failures.append(
                "missing_screening_marker"
            )
            continue

        if marker.get("passed") is True:
            continue

        summary = marker.get("summary")
        batch_tag = (
            summary.get("batch_tag")
            if isinstance(summary, dict)
            else None
        )
        technical_failures.append(
            (
                f"{batch_tag or 'unknown_batch'}:"
                f"{marker.get('status') or 'unknown_status'}:"
                f"{marker.get('error') or 'unknown_error'}"
            )
        )

    if technical_failures:
        raise CompoundScreeningTechnicalError(
            "Pre-Aggregate compound screening failed technically "
            "and was not committed. The current step stopped "
            "before Aggregate, Rank, Apply, and ACC. Resume will "
            "retry screening from the original Reflect patch. "
            "failures="
            + "; ".join(technical_failures)
        )

    return ordered_results


def run_minibatch_reflect(
    results: list[dict],
    skill_content: str,
    prediction_dir: str,
    patches_dir: str,
    workers: int,
    failure_only: bool,
    minibatch_size: int = 8,
    edit_budget: int = 4,
    random_seed: int | None = None,
    *,
    error_system: str | None = None,
    success_system: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    max_completion_tokens: int = 0,
    max_analyst_rounds: int = 3,
    use_preaggregate_compound_screening: bool = False,
    compound_screening_env: str = "forward_math",
    compound_screening_format_max_attempts: int = 3,
    compound_screening_max_completion_tokens: int = 0,
) -> list[dict | None]:
    """Full minibatch reflect stage: group → parallel optimizer calls → patches.

    Separates failure and success trajectories, splits each into minibatches
    of size M, runs all minibatches in parallel, and saves patch files.

    Parameters
    ----------
    results : list[dict]
        Rollout result dicts; see :class:`~skillopt.types.RolloutResult`.
    skill_content : str
        Current skill document.
    prediction_dir : str
        Path to ``predictions/`` with ``conversation.json`` files.
    patches_dir : str
        Path to save per-minibatch patch JSON files.
    workers : int
        Max parallel optimizer calls.
    failure_only : bool
        If True, skip success trajectories.
    minibatch_size : int
        Trajectories per group (M).
    edit_budget : int
        Max edits per minibatch (L).
    random_seed : int | None
        Optional seed used to shuffle trajectories before minibatch splitting.
    error_system, success_system : str | None
        Optional custom prompts. ``None`` = use generic defaults.

    Returns
    -------
    list[dict | None]
        Patch dicts (with ``source_type`` "failure" or "success").
    """
    os.makedirs(patches_dir, exist_ok=True)

    if use_preaggregate_compound_screening:
        if normalize_update_mode(update_mode) != "patch":
            raise ValueError(
                "Pre-Aggregate compound screening requires patch update mode."
            )
        compound_screening_env = str(
            compound_screening_env
            or "forward_math"
        )
        compound_screening_format_max_attempts = int(
            compound_screening_format_max_attempts
            or 3
        )
        if not (
            1
            <= compound_screening_format_max_attempts
            <= 3
        ):
            raise ValueError(
                "compound_screening_format_max_attempts must be between 1 and 3."
            )
        compound_screening_max_completion_tokens = max(
            0,
            int(
                compound_screening_max_completion_tokens
                or 0
            ),
        )

    # Separate failure / success
    failures = [r for r in results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9]
    successes = [r for r in results if r.get("hard")] if not failure_only else []

    failures = _shuffle_for_minibatch(failures, random_seed)
    successes = _shuffle_for_minibatch(successes, None if random_seed is None else random_seed + 1)

    # Split into minibatches
    fail_batches = _split_minibatches(failures, minibatch_size)
    succ_batches = _split_minibatches(successes, minibatch_size)

    n_fail_batches = len(fail_batches)
    n_succ_batches = len(succ_batches)
    print(
        f"    [2/6 REFLECT minibatch] "
        f"failure={len(failures)}→{n_fail_batches} groups  "
        f"success={len(successes)}→{n_succ_batches} groups  "
        f"(M={minibatch_size}, L={edit_budget}, workers={workers})"
    )

    raw_patch_entries: list[dict] = []

    # Resume support: check for already-done minibatch patches
    pending_fail: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(fail_batches):
        path = os.path.join(patches_dir, f"minibatch_fail_{idx:03d}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw_patch_entries.append(
                    {
                        "tag": f"minibatch_fail_{idx:03d}",
                        "raw_result": json.load(f),
                        "exact_minibatch_rows": batch,
                    }
                )
        else:
            pending_fail.append((idx, batch))

    pending_succ: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(succ_batches):
        path = os.path.join(patches_dir, f"minibatch_succ_{idx:03d}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw_patch_entries.append(
                    {
                        "tag": f"minibatch_succ_{idx:03d}",
                        "raw_result": json.load(f),
                        "exact_minibatch_rows": batch,
                    }
                )
        else:
            pending_succ.append((idx, batch))

    # ── Worker functions ──────────────────────────────────────────────────
    def _do_fail(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_error_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=error_system,
            step_buffer_context=step_buffer_context,
            # backward compat fallback
            rejection_context=rejection_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            max_completion_tokens=max_completion_tokens,
            max_analyst_rounds=max_analyst_rounds,
        )
        return f"minibatch_fail_{idx:03d}", patch

    def _do_succ(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_success_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=success_system,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            max_completion_tokens=max_completion_tokens,
            max_analyst_rounds=max_analyst_rounds,
        )
        return f"minibatch_succ_{idx:03d}", patch

    # Run all pending minibatches in parallel
    all_pending = (
        [("fail", idx, batch) for idx, batch in pending_fail]
        + [("succ", idx, batch) for idx, batch in pending_succ]
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, idx, batch in all_pending:
            if kind == "fail":
                futs[ex.submit(_do_fail, idx, batch)] = (kind, idx, batch)
            else:
                futs[ex.submit(_do_succ, idx, batch)] = (kind, idx, batch)

        for i, fut in enumerate(as_completed(futs), 1):
            kind, idx, batch = futs[fut]
            tag, patch = fut.result()
            if patch:
                path = os.path.join(patches_dir, f"{tag}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(patch, f, ensure_ascii=False, indent=2)
                raw_patch_entries.append(
                    {
                        "tag": tag,
                        "raw_result": patch,
                        "exact_minibatch_rows": batch,
                    }
                )
            n_edits = len(get_payload_items(patch.get("patch", {}) if patch else {}, update_mode))
            print(
                f"      [analyst] {i}/{len(all_pending)} {tag} "
                f"({len(batch)} trajs) → {n_edits} {payload_label(update_mode)}"
            )

    if not use_preaggregate_compound_screening:
        return [
            entry["raw_result"]
            for entry in raw_patch_entries
        ]

    return _run_preaggregate_compound_screening(
        raw_patch_entries,
        patches_dir=patches_dir,
        workers=workers,
        env=compound_screening_env,
        format_max_attempts=(
            compound_screening_format_max_attempts
        ),
        max_completion_tokens=(
            compound_screening_max_completion_tokens
        ),
    )
