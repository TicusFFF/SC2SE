"""Standardized I/O types for the ReflACT pipeline.

Shared dataclass definitions for the 6-stage per-step pipeline
and the 2 epoch-level stages.  All types support round-trip
conversion to/from plain dicts for incremental adoption.

Re-exports
----------
GateResult, GateAction — from skillopt.evaluation.gate
BatchSpec              — from skillopt.datasets.base
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Literal

from skillopt.evaluation.gate import GateAction, GateResult  # noqa: F401
from skillopt.datasets.base import BatchSpec  # noqa: F401
from skillopt.optimizer.structured_skills import (
    PATCH_SCHEMA_VERSION,
    AtomicSkillDraft,
    parse_patch_edit,
    parse_patch_payload,
)


# ── Atomic types ─────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """One outer edit proposal operating on atomic solver rules."""

    op: EditOp
    skills: list[AtomicSkillDraft] = field(default_factory=list)
    target_skill_id: str = ""
    support_count: int | None = None
    source_type: str | None = None
    merge_level: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        canonical = parse_patch_edit(d, path="$.edit")
        return cls(
            op=canonical["op"],
            skills=[
                AtomicSkillDraft(**skill)
                for skill in canonical.get("skills", [])
            ],
            target_skill_id=canonical.get("target_skill_id", ""),
            support_count=canonical.get("support_count"),
            source_type=canonical.get("source_type"),
            merge_level=canonical.get("merge_level"),
        )

    def to_dict(self) -> dict:
        raw: dict[str, Any] = {"op": self.op}

        if self.target_skill_id:
            raw["target_skill_id"] = self.target_skill_id

        if self.op != "delete":
            raw["skills"] = [
                skill.to_dict() for skill in self.skills
            ]

        for name in ("support_count", "source_type", "merge_level"):
            value = getattr(self, name)
            if value is not None:
                raw[name] = value

        return parse_patch_edit(raw, path="$.edit")


@dataclass
class Patch:
    """A set of edits with reasoning.

    Output of Aggregate (③), Select (④); input to Update (⑤).
    """

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None
    evidence_sidecar: dict[str, list[dict[str, str]]] | None = None
    schema_version: str = PATCH_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        trusted: bool = False,
    ) -> Patch:
        canonical = parse_patch_payload(
            d,
            allow_ranking_details=True,
            allow_trusted_metadata=trusted,
        )
        return cls(
            edits=[Edit.from_dict(edit) for edit in canonical["edits"]],
            reasoning=canonical["reasoning"],
            ranking_details=canonical.get("ranking_details"),
            evidence_sidecar=canonical.get("_evidence_sidecar"),
            schema_version=canonical["schema_version"],
        )

    @classmethod
    def from_trusted_dict(cls, d: dict) -> Patch:
        """Parse an internal patch whose evidence sidecar came from code."""

        return cls.from_dict(d, trusted=True)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "reasoning": self.reasoning,
            "edits": [
                edit.to_dict() if isinstance(edit, Edit) else edit
                for edit in self.edits
            ],
        }
        if self.ranking_details is not None:
            d["ranking_details"] = self.ranking_details
        if self.evidence_sidecar is not None:
            d["_evidence_sidecar"] = self.evidence_sidecar
        return parse_patch_payload(
            d,
            allow_ranking_details=True,
            allow_trusted_metadata=True,
        )


# ── Stage ① ROLLOUT ──────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Result of a single episode/task rollout.

    Universal fields are required; env-specific fields live in ``extras``.
    """

    id: str
    hard: int
    soft: float
    n_turns: int = 0
    fail_reason: str = ""
    task_type: str = ""
    task_description: str = ""
    predicted_answer: str = ""
    question: str = ""
    reference_text: str = ""
    target_system_prompt: str = ""
    target_user_prompt: str = ""
    spreadsheet_preview: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False,  # type: ignore[assignment]
    )

    @classmethod
    def _get_known_fields(cls) -> frozenset[str]:
        if cls._KNOWN_FIELDS is None:
            cls._KNOWN_FIELDS = frozenset(
                f.name for f in dc_fields(cls)
                if f.name != "_KNOWN_FIELDS"
            )
        return cls._KNOWN_FIELDS

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        known = cls._get_known_fields()
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            n_turns=int(d.get("n_turns", 0)),
            fail_reason=str(d.get("fail_reason", "")),
            task_type=str(d.get("task_type", "")),
            task_description=str(d.get("task_description", "")),
            predicted_answer=str(d.get("predicted_answer", "")),
            question=str(d.get("question", "")),
            reference_text=str(d.get("reference_text", "")),
            target_system_prompt=str(d.get("target_system_prompt", "")),
            target_user_prompt=str(d.get("target_user_prompt", "")),
            spreadsheet_preview=str(d.get("spreadsheet_preview", "")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
        }
        for attr in (
            "n_turns", "fail_reason", "task_type", "task_description",
            "predicted_answer", "question", "reference_text",
            "target_system_prompt", "target_user_prompt",
            "spreadsheet_preview",
        ):
            val = getattr(self, attr)
            if val:
                d[attr] = val
        d.update(self.extras)
        return d


# ── Stage ② REFLECT ──────────────────────────────────────────────────────

@dataclass
class FailureSummaryEntry:
    """One entry in the failure summary produced by error analysts."""

    failure_type: str
    count: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FailureSummaryEntry:
        return cls(
            failure_type=d.get("failure_type", ""),
            count=int(d.get("count", 0)),
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "description": self.description,
        }


@dataclass
class RawPatch:
    """Analyst output from the Reflect stage — a patch with provenance.

    Wraps the dict produced by ``run_error_analyst_minibatch``
    and ``run_success_analyst_minibatch``.
    """

    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0
    failure_summary: list[FailureSummaryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> RawPatch | None:
        if d is None:
            return None
        inner = d.get("patch", d)
        if not isinstance(inner, dict):
            return None
        patch = Patch.from_dict(inner)
        return cls(
            patch=patch,
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0)),
            failure_summary=[
                FailureSummaryEntry.from_dict(fs)
                for fs in d.get("failure_summary", [])
            ],
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "patch": self.patch.to_dict(),
            "source_type": self.source_type,
            "batch_size": self.batch_size,
        }
        if self.failure_summary:
            d["failure_summary"] = [fs.to_dict() for fs in self.failure_summary]
        return d


# ── Epoch-level: SLOW_UPDATE ─────────────────────────────────────────────

@dataclass
class SlowUpdateResult:
    """Output of the epoch-level slow update stage (EMA / regularization)."""

    reasoning: str = ""
    slow_update_content: str = ""
    action: str = ""
    time_s: float | None = None
    prev_hard: float | None = None
    curr_hard: float | None = None
    selection_hard: float | None = None
    selection_soft: float | None = None
    candidate_hash: str = ""
    update_origin: str = ""
    update_target: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> SlowUpdateResult | None:
        if d is None:
            return None
        return cls(
            reasoning=d.get("reasoning", ""),
            slow_update_content=d.get("slow_update_content", ""),
            action=d.get("action", ""),
            time_s=d.get("time_s"),
            prev_hard=d.get("prev_hard"),
            curr_hard=d.get("curr_hard"),
            selection_hard=d.get("selection_hard"),
            selection_soft=d.get("selection_soft"),
            candidate_hash=d.get("candidate_hash", ""),
            update_origin=d.get("update_origin", ""),
            update_target=d.get("update_target", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "slow_update_content": self.slow_update_content,
        }
        if self.action:
            d["action"] = self.action
        if self.time_s is not None:
            d["time_s"] = self.time_s
        if self.prev_hard is not None:
            d["prev_hard"] = self.prev_hard
        if self.curr_hard is not None:
            d["curr_hard"] = self.curr_hard
        if self.selection_hard is not None:
            d["selection_hard"] = self.selection_hard
        if self.selection_soft is not None:
            d["selection_soft"] = self.selection_soft
        if self.candidate_hash:
            d["candidate_hash"] = self.candidate_hash
        if self.update_origin:
            d["update_origin"] = self.update_origin
        if self.update_target:
            d["update_target"] = self.update_target
        return d
