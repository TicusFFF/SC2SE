"""Conservative pairwise functional-equivalence judgment for Skill documents."""
from __future__ import annotations

from typing import Any

from skillopt.model import chat_optimizer
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


_REQUIRED_OUTPUT_KEYS = {
    "equivalent",
    "reason",
}

_ALLOWED_DECISIONS = {
    "YES",
    "NO",
}


def build_skill_equivalence_user_prompt(
    *,
    historical_skill: str,
    candidate_skill: str,
) -> str:
    """Build one historical-Skill versus one candidate-Skill comparison."""
    return (
        "## Historical Skill\n\n"
        "----- BEGIN HISTORICAL SKILL -----\n"
        f"{str(historical_skill).rstrip()}\n"
        "----- END HISTORICAL SKILL -----\n\n"
        "## Candidate Skill\n\n"
        "----- BEGIN CANDIDATE SKILL -----\n"
        f"{str(candidate_skill).rstrip()}\n"
        "----- END CANDIDATE SKILL -----\n\n"
        "Decide whether these two complete Skill documents are "
        "functionally equivalent."
    )


def _safe_no_judgment(
    *,
    raw_response: str = "",
    usage: dict | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Fall back conservatively to normal Forward evaluation."""
    result: dict[str, Any] = {
        "is_equivalent": False,
        "decision": "NO",
        "reason": "",
        "valid_response": False,
        "raw_response": raw_response,
        "usage": usage if isinstance(usage, dict) else {},
    }

    if error:
        result["error"] = error

    return result


def judge_skill_equivalence(
    *,
    historical_skill: str,
    candidate_skill: str,
    env: str = "forward_math",
    max_completion_tokens: int = 0,
) -> dict[str, Any]:
    """
    Conservatively judge one historical Skill and one candidate Skill.

    Any API failure, malformed output, unexpected field, missing reason, or
    uncertain judgment becomes NO, so the original Forward evaluation runs.
    """
    user_prompt = build_skill_equivalence_user_prompt(
        historical_skill=historical_skill,
        candidate_skill=candidate_skill,
    )

    try:
        response, usage = chat_optimizer(
            system=load_prompt(
                "skill_equivalence",
                env=env,
            ),
            user=user_prompt,
            max_completion_tokens=max_completion_tokens,
            retries=4,
            stage="skill_equivalence",
        )

    except Exception as error:  # noqa: BLE001
        return _safe_no_judgment(
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    text = str(response).strip()
    payload = extract_json(text)

    # Only one bare JSON object with exactly the required fields is accepted.
    is_bare_json_object = (
        text.startswith("{")
        and text.endswith("}")
    )

    valid_response = (
        is_bare_json_object
        and isinstance(payload, dict)
        and set(payload) == _REQUIRED_OUTPUT_KEYS
        and isinstance(
            payload.get("equivalent"),
            str,
        )
        and payload.get("equivalent") in _ALLOWED_DECISIONS
        and isinstance(
            payload.get("reason"),
            str,
        )
        and bool(
            payload.get("reason", "").strip()
        )
    )

    if not valid_response:
        return _safe_no_judgment(
            raw_response=text,
            usage=usage,
        )

    decision = payload["equivalent"]
    reason = payload["reason"].strip()

    return {
        "is_equivalent": decision == "YES",
        "decision": decision,
        "reason": reason,
        "valid_response": True,
        "raw_response": text,
        "usage": usage if isinstance(usage, dict) else {},
    }


def find_equivalent_historical_skill(
    *,
    candidate_skill: str,
    historical_skills: list[dict[str, Any]],
    env: str = "forward_math",
    max_completion_tokens: int = 0,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    """
    Compare newest-to-oldest, one historical Skill at a time.

    Stop at the first conservative YES. Historical identifiers are attached
    by controller code after the model call and are never shown to the model.
    """
    checks: list[dict[str, Any]] = []

    for historical in historical_skills:
        judgment = judge_skill_equivalence(
            historical_skill=str(
                historical["skill"]
            ),
            candidate_skill=candidate_skill,
            env=env,
            max_completion_tokens=max_completion_tokens,
        )

        checks.append({
            "historical_step": int(
                historical["step"]
            ),
            "historical_hash": str(
                historical["hash"]
            ),
            "historical_skill_path": str(
                historical["skill_path"]
            ),
            **judgment,
        })

        if judgment["is_equivalent"]:
            return historical, checks

    return None, checks
