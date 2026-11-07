from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from skillopt.optimizer.structured_skills import (
    build_solver_projection,
    parse_skill_library_text,
)


def render_skill_section(
    skill_content: str,
) -> str:
    canonical_json = (
        skill_content or ""
    ).strip()

    if not canonical_json:
        return ""

    library = parse_skill_library_text(
        canonical_json
    )
    projection = build_solver_projection(
        library
    ).strip()

    if not projection:
        return ""

    return (
        "## Current Skill\n"
        + projection
        + "\n"
    )


def load_rollout_system(
    prompt_path: str | Path,
) -> str:
    prompt_path = Path(prompt_path)

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Cannot find rollout system prompt: {prompt_path}"
        )

    return prompt_path.read_text(
        encoding="utf-8"
    )


def build_a0_messages(
    *,
    problem: str,
    skill_content: str,
    rollout_system_template: str,
) -> List[Dict[str, str]]:
    """
    SkillOpt-style A0 prompt.

    System:
      rollout_system.md with {skill_section}

    User:
      current problem only
    """
    problem = str(problem or "").strip()

    if not problem:
        raise ValueError(
            "A0 requires a nonempty problem."
        )

    system_prompt = rollout_system_template.format(
        skill_section=render_skill_section(
            skill_content
        )
    )

    user_prompt = (
        "## Problem\n"
        + problem
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
