from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ForwardStepEvidence:
    V_f: str
    r_f: bool
    R_f: str
    F_f: Optional[str] = None
    y_prime: Optional[str] = None
    a_prime: Optional[str] = None


@dataclass
class ForwardRolloutEvidence:
    """
    Rollout Evidence tau.

    Base:
      tau = {y, a, V_f, r_f, R_f}

    Failure:
      + F_f

    Repair:
      + y'
    """

    sample_id: str
    q: str
    skill_version: str
    y: str
    a: str
    steps: List[ForwardStepEvidence]
    final_response: Optional[str] = None
    final_answer: Optional[str] = None
    verification_unavailable_reason: str = ""

    def initial_pass(self) -> bool:
        return bool(
            self.steps
            and self.steps[0].r_f
        )

    def forward_final_pass(self) -> bool:
        return any(
            step.r_f
            for step in self.steps
        )

    def has_failure_feedback(self) -> bool:
        return any(
            step.F_f
            for step in self.steps
        )

    def has_repair(self) -> bool:
        return any(
            step.y_prime
            for step in self.steps
        )

    def trajectory_type(self) -> str:
        if self.verification_unavailable_reason:
            return "verifier_error"

        if self.initial_pass():
            return "passed"

        if self.has_repair():
            return "failed_repaired"

        if self.has_failure_feedback():
            return "failed"

        return "verifier_error"

    def hard_score(self) -> int:
        """
        Optimize initial-answer quality.

        A15 repair is retained as trajectory evidence, but it must not
        turn hard into 1 when the initial response failed.
        """
        return 1 if self.initial_pass() else 0

    def soft_score(self) -> float:
        return float(
            self.hard_score()
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evidence_to_conversation(
    evidence: ForwardRolloutEvidence,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "evidence_priority": "protected",
            "content": (
                "[Problem]\n"
                f"{evidence.q}"
            ),
        },
        {
            "role": "assistant",
            "evidence_priority": "compressible",
            "content": (
                "[Initial Response]\n"
                f"{evidence.y}"
            ),
        },
        {
            "role": "assistant",
            "evidence_priority": "protected",
            "content": (
                "[Initial Extracted Answer]\n"
                f"{evidence.a}"
            ),
        },
    ]

    for index, step in enumerate(
        evidence.steps,
        start=1,
    ):
        protected_parts = [
            f"[Forward Verification Step {index} Core]",
            "",
            "[Forward Verdict r_f]",
            str(step.r_f),
        ]

        if step.F_f:
            protected_parts.extend(
                [
                    "",
                    "[Forward Feedback Summary F_f]",
                    step.F_f,
                ]
            )

        if step.a_prime:
            protected_parts.extend(
                [
                    "",
                    "[Revised Answer a']",
                    step.a_prime,
                ]
            )

        messages.append(
            {
                "role": "system",
                "evidence_priority": "protected",
                "content": "\n".join(
                    protected_parts
                ),
            }
        )

        detail_parts = [
            f"[Forward Verification Step {index} Details]",
            "",
            "[Verification Flow V_f]",
            step.V_f,
            "",
            "[Execution Record R_f]",
            step.R_f,
        ]

        if step.y_prime:
            detail_parts.extend(
                [
                    "",
                    "[Revised Response y']",
                    step.y_prime,
                ]
            )

        messages.append(
            {
                "role": "system",
                "evidence_priority": "compressible",
                "content": "\n".join(
                    detail_parts
                ),
            }
        )

    revised_responses = {
        str(step.y_prime).strip()
        for step in evidence.steps
        if step.y_prime
    }
    final_response = str(
        evidence.final_response or ""
    ).strip()
    if (
        final_response
        and final_response != str(evidence.y).strip()
        and final_response not in revised_responses
    ):
        messages.append(
            {
                "role": "assistant",
                "evidence_priority": "compressible",
                "content": (
                    "[Final Response]\n"
                    f"{final_response}"
                ),
            }
        )

    if (
        evidence.final_answer is not None
        or evidence.steps
        or evidence.verification_unavailable_reason
    ):
        messages.append(
            {
                "role": "assistant",
                "evidence_priority": "protected",
                "content": (
                    "[Final Extracted Answer]\n"
                    f"{evidence.final_answer or ''}\n\n"
                    "[Initial Forward Verdict]\n"
                    f"{evidence.initial_pass()}\n\n"
                    "[Final Forward Verdict]\n"
                    f"{evidence.forward_final_pass()}\n\n"
                    "[Verification Unavailable Reason]\n"
                    f"{evidence.verification_unavailable_reason}"
                ),
            }
        )

    return messages


def evidence_to_skillopt_result(
    evidence: ForwardRolloutEvidence,
) -> Dict[str, Any]:
    return {
        "id": evidence.sample_id,
        "task_description": evidence.q,
        "task_type": "forward_math",
        "hard": evidence.hard_score(),
        "soft": evidence.soft_score(),
        "trajectory_type": evidence.trajectory_type(),
        "initial_response": evidence.y,
        "initial_answer": evidence.a,
        "forward_initial_pass": evidence.initial_pass(),
        "forward_final_pass": evidence.forward_final_pass(),
        "forward_repair_attempted": (
            evidence.has_repair()
            and not evidence.initial_pass()
        ),
        "forward_repair_accepted": (
            evidence.forward_final_pass()
            and not evidence.initial_pass()
        ),
        "forward_steps": [
            asdict(step)
            for step in evidence.steps
        ],
        "final_response": evidence.final_response,
        "final_answer": evidence.final_answer,
        "verification_unavailable_reason": (
            evidence.verification_unavailable_reason
        ),
        "fail_reason": build_fail_reason(evidence),
    }


def build_fail_reason(
    evidence: ForwardRolloutEvidence,
) -> str:
    if evidence.verification_unavailable_reason:
        return (
            "Forward verification unavailable: "
            + evidence.verification_unavailable_reason
        )

    if evidence.initial_pass():
        return ""

    for step in evidence.steps:
        if step.F_f:
            return (
                "Forward verification failed: "
                + step.F_f
            )

    return (
        "Forward verification failed or "
        "could not be completed."
    )
