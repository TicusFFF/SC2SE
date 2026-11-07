from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from S5_bridge.S5_utils import S5_bool
from S5_bridge.data_sanitizer import (
    validate_sample_id,
)
from S5_bridge.forward_evidence import (
    ForwardRolloutEvidence,
    ForwardStepEvidence,
)


ChatFn = Callable[
    [List[Dict[str, str]]],
    str,
]

AnswerExtractor = Callable[
    [str, str],
    str,
]


class ForwardSkillRunner:
    """
    Thin wrapper around the original Forward project.

    It does not change A8/A11/A12/A13/A14/A15/A16/A17.
    It only supplies A0-with-skill output and collects S5 trace.
    """

    def __init__(
        self,
        *,
        progco_agent: Any,
        a0_chat_fn: ChatFn,
        answer_extractor: AnswerExtractor,
        skill_version: str = "S_t",
    ):
        self.progco_agent = progco_agent
        self.a0_chat_fn = a0_chat_fn
        self.answer_extractor = answer_extractor
        self.skill_version = skill_version

    def _extract_answer(
        self,
        query: str,
        response: str,
    ) -> str:
        """
        Reuse Forward A8's query-aware extractor.

        Do not catch TypeError here: a TypeError raised inside A8 should
        remain visible rather than being mistaken for an alternative
        one-argument signature.
        """
        return str(
            self.answer_extractor(
                query,
                response,
            )
        )

    def run_one_with_messages(
        self,
        *,
        item: Dict[str, Any],
        a0_messages: List[Dict[str, str]],
        skill_content: str,
        initial_response_override: Optional[Dict[str, Any]] = None,
        student_only: bool = False,
        student_forward_verification_only: bool = False,
    ) -> ForwardRolloutEvidence:
        del skill_content

        sample_id = validate_sample_id(
            item["id"]
        )

        q = (
            item.get("q")
            or item.get("query")
            or item.get("problem")
        )

        if not q:
            raise KeyError(
                f"Missing problem text for sample {sample_id}"
            )

        if initial_response_override is not None:
            y = str(
                initial_response_override.get("response")
                or ""
            ).strip()
            a = str(
                initial_response_override.get("pred")
                or ""
            ).strip()
        else:
            y = str(
                self.a0_chat_fn(a0_messages)
                or ""
            ).strip()

        if not y:
            response_source = (
                "shared initial response"
                if initial_response_override is not None
                else "A0"
            )
            raise RuntimeError(
                f"{response_source} returned an empty response "
                f"for sample {sample_id}"
            )

        self.last_initial_response = y
        self.last_initial_answer = ""
        self.last_answer_extraction_fallback = ""

        if initial_response_override is None:
            a = self._extract_answer(q, y)

            if (
                (
                    student_only
                    or student_forward_verification_only
                )
                and not str(a).strip()
            ):
                a = y
                self.last_answer_extraction_fallback = (
                    "full_response"
                )

        self.last_initial_answer = a

        # if not str(a).strip():
        #     answer_source = (
        #         "shared initial pred"
        #         if initial_response_override is not None
        #         else "A8"
        #     )
        #     raise RuntimeError(
        #         f"{answer_source} returned an empty answer "
        #         f"for sample {sample_id}"
        #     )

        if (
            initial_response_override is None
            and not str(a).strip()
        ):
            raise RuntimeError(
                f"A8 returned an empty answer "
                f"for sample {sample_id}"
            )

        if student_only:
            return ForwardRolloutEvidence(
                sample_id=sample_id,
                q=q,
                skill_version=self.skill_version,
                y=y,
                a=str(a),
                steps=[],
                final_response=y,
                final_answer=str(a),
            )

        first_response_override = {
            "response": y,
            "pred": a,
        }

        sample = dict(item)
        sample.update(
            {
                "id": sample_id,
                "q": q,
                "problem": q,
                "query": q,
                "_S5_first_response": first_response_override,
            }
        )

        executor = getattr(
            self.progco_agent,
            "llm_code_executor",
            None,
        )

        if (
            executor is not None
            and hasattr(executor, "sample_id")
        ):
            executor.sample_id = sample_id

        inference_kwargs: Dict[str, Any] = {
            "first_response_override": first_response_override,
            "return_S5_trace": True,
        }
        if student_forward_verification_only:
            inference_kwargs[
                "stop_after_first_verification"
            ] = True

        result = self.progco_agent.inference(
            sample,
            **inference_kwargs,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "Forward inference result must be a dict."
            )

        S5 = result.get("S5")

        if not isinstance(S5, dict):
            raise RuntimeError(
                "Forward did not return output['S5']. "
                "Verify that return_S5_trace=True is supported."
            )

        trace = S5.get("trace", [])

        if not isinstance(trace, list):
            raise TypeError(
                "Forward output['S5']['trace'] must be a list."
            )

        steps: list[ForwardStepEvidence] = []

        for step in trace:
            if not isinstance(step, dict):
                raise TypeError(
                    "Each Forward S5 trace step must be a dict."
                )

            steps.append(
                ForwardStepEvidence(
                    V_f=str(
                        step.get("V_f", "")
                    ),
                    r_f=S5_bool(
                        step.get("r_f", False)
                    ),
                    R_f=str(
                        step.get("R_f", "")
                    ),
                    F_f=step.get("F_f"),
                    y_prime=step.get("y_prime"),
                    a_prime=step.get("a_prime"),
                )
            )

        if not steps:
            raise RuntimeError(
                f"Forward returned an empty S5 trace for sample {sample_id}"
            )

        initial_response = S5.get(
            "initial_response"
        )
        if initial_response is None:
            initial_response = y

        initial_answer = S5.get(
            "initial_answer"
        )
        if initial_answer is None:
            initial_answer = a

        verification_unavailable_reason = str(
            S5.get(
                "verification_unavailable_reason",
                "",
            )
            or ""
        ).strip()

        return ForwardRolloutEvidence(
            sample_id=sample_id,
            q=q,
            skill_version=self.skill_version,
            y=str(initial_response),
            a=str(initial_answer),
            steps=steps,
            final_response=S5.get(
                "final_response"
            ),
            final_answer=S5.get(
                "final_answer"
            ),
            verification_unavailable_reason=(
                verification_unavailable_reason
            ),
        )
