"""Canonical JSON contracts for S5-7-8 atomic solver skills.

The main skill library contains only ordered atomic skill records.  Model
authored patches contain only the two semantic fields used by the initial
mathematics solver: ``title`` and ``solver_guidance``.  Stable IDs, support
counts, and evidence are controlled by trusted Python code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping


SKILL_LIBRARY_SCHEMA_VERSION = "s5_7_8_atomic_solver_skill_library_v1"
PATCH_SCHEMA_VERSION = "s5_7_8_atomic_solver_skill_patch_v1"
SOLVER_VIEW_SCHEMA_VERSION = "s5_7_8_solver_skill_view_v1"

SKILL_ID_PREFIX = "skill_sha256_"
SKILL_ID_RE = re.compile(r"^skill_sha256_[0-9a-f]{64}$")
ROLE_FIELDS = ("title", "solver_guidance")
PATCH_OPS = frozenset({"append", "insert_after", "replace", "delete"})
SOURCE_TYPES = frozenset({"failure", "success"})

PatchOp = Literal["append", "insert_after", "replace", "delete"]
SourceType = Literal["failure", "success"]
SkillLibrary = dict[str, Any]
AtomicSkillRecord = dict[str, Any]


class SkillJsonError(ValueError):
    """Path-aware fail-closed validation error for skill JSON."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = str(code)
        self.path = str(path)
        self.message = str(message)
        super().__init__(f"{self.code} at {self.path}: {self.message}")


StructuredSkillError = SkillJsonError


def _fail(code: str, path: str, message: str) -> None:
    raise SkillJsonError(code, path, message)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", "$", f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite numbers."""

    if not isinstance(text, str):
        _fail("invalid_json_input", "$", "JSON input must be a string.")

    def reject_constant(value: str) -> None:
        _fail("non_finite_number", "$", f"JSON constant {value!r} is not allowed.")

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=reject_constant,
        )
    except SkillJsonError:
        raise
    except json.JSONDecodeError as error:
        _fail("invalid_json", "$", str(error))


def canonical_json_text(value: Any) -> str:
    """Return the canonical compact JSON representation used for hashing."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("non_json_value", "$", str(error))


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pretty_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_type", path, "Expected a JSON object.")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("invalid_type", path, "Expected a JSON array.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    path: str,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing:
        _fail("missing_fields", path, f"Missing fields: {missing}.")
    if extra:
        _fail("unexpected_fields", path, f"Unexpected fields: {extra}.")


def _normalise_text(value: Any, path: str, *, one_line: bool = False) -> str:
    if not isinstance(value, str):
        _fail("invalid_type", path, "Expected a string.")
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        _fail("empty_text", path, "Text must be non-empty.")
    for character in text:
        if ord(character) < 32 and character not in {"\n", "\t"}:
            _fail("control_character", path, "Unsupported control character.")
    if one_line and "\n" in text:
        _fail("multiline_text", path, "Expected one line.")
    if one_line:
        text = re.sub(r"[ \t]+", " ", text)
    return text


@dataclass(frozen=True, slots=True)
class AtomicSkillDraft:
    title: str
    solver_guidance: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "solver_guidance": self.solver_guidance,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    evidence_id: str
    source_type: SourceType

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
        }


def validate_skill_draft(value: Any, *, path: str = "$.skill") -> dict[str, str]:
    """Validate and canonicalize the two model-authored semantic fields."""

    if isinstance(value, AtomicSkillDraft):
        value = value.to_dict()
    raw = _require_mapping(value, path)
    _require_exact_keys(raw, required=set(ROLE_FIELDS), path=path)
    return {
        "title": _normalise_text(raw["title"], f"{path}.title", one_line=True),
        "solver_guidance": _normalise_text(
            raw["solver_guidance"], f"{path}.solver_guidance"
        ),
    }


def parse_atomic_skill_draft(
    value: Any,
    *,
    path: str = "$.skill",
) -> AtomicSkillDraft:
    return AtomicSkillDraft(**validate_skill_draft(value, path=path))


def normalise_atomic_skill_draft(draft: AtomicSkillDraft) -> AtomicSkillDraft:
    return parse_atomic_skill_draft(draft)


def semantic_payload(draft: AtomicSkillDraft | Mapping[str, Any]) -> dict[str, str]:
    return validate_skill_draft(draft)


def skill_id_from_draft(draft: AtomicSkillDraft | Mapping[str, Any]) -> str:
    """Derive an immutable full-SHA256 ID from semantic fields only."""

    return SKILL_ID_PREFIX + payload_hash(validate_skill_draft(draft))


compute_skill_id = skill_id_from_draft


def _validate_evidence_ref(value: Any, *, path: str) -> dict[str, str]:
    if isinstance(value, EvidenceRef):
        value = value.to_dict()
    raw = _require_mapping(value, path)
    _require_exact_keys(raw, required={"evidence_id", "source_type"}, path=path)
    evidence_id = _normalise_text(
        raw["evidence_id"], f"{path}.evidence_id", one_line=True
    )
    source_type = raw["source_type"]
    if source_type not in SOURCE_TYPES:
        _fail(
            "invalid_source_type",
            f"{path}.source_type",
            "Expected 'failure' or 'success'.",
        )
    return {"evidence_id": evidence_id, "source_type": source_type}


def normalise_evidence(
    refs: Iterable[EvidenceRef | Mapping[str, Any]],
) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for index, value in enumerate(refs):
        ref = _validate_evidence_ref(value, path=f"$.evidence[{index}]")
        previous = by_id.get(ref["evidence_id"])
        if previous is not None and previous["source_type"] != ref["source_type"]:
            _fail(
                "conflicting_evidence_source",
                f"$.evidence[{index}]",
                f"Evidence {ref['evidence_id']!r} has conflicting source types.",
            )
        by_id[ref["evidence_id"]] = ref
    return [by_id[key] for key in sorted(by_id)]


def build_atomic_skill_record(
    draft: AtomicSkillDraft | Mapping[str, Any],
    evidence: Iterable[EvidenceRef | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    canonical_draft = validate_skill_draft(draft)
    canonical_evidence = normalise_evidence(evidence)
    return {
        "skill_id": skill_id_from_draft(canonical_draft),
        **canonical_draft,
        "support_count": len(canonical_evidence),
        "evidence": canonical_evidence,
    }


def parse_atomic_skill_record(value: Any, *, path: str) -> dict[str, Any]:
    raw = _require_mapping(value, path)
    required = {"skill_id", *ROLE_FIELDS, "support_count", "evidence"}
    _require_exact_keys(raw, required=required, path=path)
    draft = validate_skill_draft(
        {field: raw[field] for field in ROLE_FIELDS},
        path=path,
    )
    skill_id = raw["skill_id"]
    if not isinstance(skill_id, str) or not SKILL_ID_RE.fullmatch(skill_id):
        _fail(
            "invalid_skill_id",
            f"{path}.skill_id",
            "Expected a full SHA-256 skill ID.",
        )
    expected = skill_id_from_draft(draft)
    if skill_id != expected:
        _fail("skill_id_mismatch", f"{path}.skill_id", f"Expected {expected!r}.")
    support_count = raw["support_count"]
    if type(support_count) is not int or support_count < 0:
        _fail(
            "invalid_support_count",
            f"{path}.support_count",
            "Expected a non-negative integer.",
        )
    evidence_raw = _require_list(raw["evidence"], f"{path}.evidence")
    evidence = normalise_evidence(
        _validate_evidence_ref(item, path=f"{path}.evidence[{index}]")
        for index, item in enumerate(evidence_raw)
    )
    if support_count != len(evidence):
        _fail(
            "support_count_mismatch",
            f"{path}.support_count",
            f"Expected {len(evidence)} from distinct evidence IDs.",
        )
    return {
        "skill_id": skill_id,
        **draft,
        "support_count": support_count,
        "evidence": evidence,
    }


def merge_verified_evidence(
    record: Mapping[str, Any],
    refs: Iterable[EvidenceRef | Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = parse_atomic_skill_record(record, path="$.skill")
    return build_atomic_skill_record(
        {field: canonical[field] for field in ROLE_FIELDS},
        [*canonical["evidence"], *list(refs)],
    )


def empty_skill_library() -> dict[str, Any]:
    return {
        "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
        "skills": [],
    }


def validate_skill_library(value: Any) -> dict[str, Any]:
    """Return a canonical atomic solver-skill library or fail closed."""

    raw = _require_mapping(value, "$")
    _require_exact_keys(raw, required={"schema_version", "skills"}, path="$")
    if raw["schema_version"] != SKILL_LIBRARY_SCHEMA_VERSION:
        _fail(
            "schema_version_mismatch",
            "$.schema_version",
            f"Expected {SKILL_LIBRARY_SCHEMA_VERSION!r}.",
        )
    skills_raw = _require_list(raw["skills"], "$.skills")
    skills = [
        parse_atomic_skill_record(item, path=f"$.skills[{index}]")
        for index, item in enumerate(skills_raw)
    ]
    skill_ids = [item["skill_id"] for item in skills]
    if len(skill_ids) != len(set(skill_ids)):
        _fail("duplicate_skill_id", "$.skills", "Skill IDs must be unique.")
    return {
        "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
        "skills": skills,
    }


def parse_skill_library_text(text: str) -> dict[str, Any]:
    return validate_skill_library(strict_json_loads(text))


def parse_skill_library(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return parse_skill_library_text(value)
    return validate_skill_library(value)


def dump_skill_library(library: Mapping[str, Any]) -> str:
    return canonical_json_text(validate_skill_library(library))


serialise_skill_library = dump_skill_library


def skill_library_to_dict(library: Mapping[str, Any]) -> dict[str, Any]:
    return validate_skill_library(library)


def compute_library_hash(library: Mapping[str, Any]) -> str:
    return payload_hash(validate_skill_library(library))


def skill_behavior_hash(library: Mapping[str, Any]) -> str:
    """Hash ordered solver semantics, excluding evidence and counters."""

    canonical = validate_skill_library(library)
    semantics = [
        {field: skill[field] for field in ROLE_FIELDS}
        for skill in canonical["skills"]
    ]
    return payload_hash(semantics)


def make_active_skill_library(
    skills: Iterable[Mapping[str, Any]],
    **_unused: Any,
) -> dict[str, Any]:
    return validate_skill_library({
        "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
        "skills": list(skills),
    })


def parse_patch_edit(
    value: Any,
    *,
    path: str,
    known_skill_ids: set[str] | None = None,
) -> dict[str, Any]:
    raw = dict(_require_mapping(value, path))
    metadata = {}

    for name in ("support_count", "source_type", "merge_level"):
        if name not in raw:
            continue

        item = raw.pop(name)

        if name in {"support_count", "merge_level"}:
            if type(item) is not int or item < 0:
                _fail(
                    "invalid_proposal_metadata",
                    f"{path}.{name}",
                    "Expected a non-negative integer.",
                )
        elif item not in {"failure", "success"}:
            _fail(
                "invalid_proposal_metadata",
                f"{path}.{name}",
                "Expected failure or success.",
            )

        metadata[name] = item
    op = raw.get("op")
    if op not in PATCH_OPS:
        _fail("invalid_patch_op", f"{path}.op", f"Expected one of {sorted(PATCH_OPS)}.")
    if op == "append":
        _require_exact_keys(raw, required={"op", "skills"}, path=path)
    elif op in {"insert_after", "replace"}:
        _require_exact_keys(raw, required={"op", "target_skill_id", "skills"}, path=path)
    else:
        _require_exact_keys(raw, required={"op", "target_skill_id"}, path=path)

    result: dict[str, Any] = {"op": op}
    if op != "append":
        target = raw["target_skill_id"]
        if not isinstance(target, str) or not SKILL_ID_RE.fullmatch(target):
            _fail(
                "invalid_target_skill_id",
                f"{path}.target_skill_id",
                "Expected a full SHA-256 skill ID.",
            )
        if known_skill_ids is not None and target not in known_skill_ids:
            _fail(
                "unknown_target_skill_id",
                f"{path}.target_skill_id",
                f"Unknown target {target!r}.",
            )
        result["target_skill_id"] = target
    if op != "delete":
        skills_raw = _require_list(raw["skills"], f"{path}.skills")
        if not skills_raw:
            _fail("empty_edit_skills", f"{path}.skills", "A non-delete Edit needs at least one skill.")
        if op == "replace" and len(skills_raw) != 1:
            _fail(
                "replace_requires_one_skill",
                f"{path}.skills",
                "A replace Edit must contain exactly one replacement skill.",
            )
        skills = [
            validate_skill_draft(item, path=f"{path}.skills[{index}]")
            for index, item in enumerate(skills_raw)
        ]
        ids = [skill_id_from_draft(item) for item in skills]
        if len(ids) != len(set(ids)):
            _fail("duplicate_edit_skill", f"{path}.skills", "One Edit cannot repeat a skill.")
        result["skills"] = skills
    result.update(metadata)
    return result


def parse_patch_payload(
    value: str | Mapping[str, Any],
    *,
    known_skill_ids: set[str] | None = None,
    allow_ranking_details: bool = True,
    allow_trusted_metadata: bool = False,
) -> dict[str, Any]:
    if isinstance(value, str):
        value = strict_json_loads(value)
    raw = _require_mapping(value, "$")
    optional = {"ranking_details"} if allow_ranking_details else set()
    if allow_trusted_metadata:
        optional.add("_evidence_sidecar")
    _require_exact_keys(
        raw,
        required={"schema_version", "reasoning", "edits"},
        optional=optional,
        path="$",
    )
    if raw["schema_version"] != PATCH_SCHEMA_VERSION:
        _fail(
            "schema_version_mismatch",
            "$.schema_version",
            f"Expected {PATCH_SCHEMA_VERSION!r}.",
        )
    if not isinstance(raw["reasoning"], str):
        _fail("invalid_reasoning", "$.reasoning", "Expected a string.")
    edits_raw = _require_list(raw["edits"], "$.edits")
    edits = [
        parse_patch_edit(
            item,
            path=f"$.edits[{index}]",
            known_skill_ids=known_skill_ids,
        )
        for index, item in enumerate(edits_raw)
    ]
    patch_skill_ids = [
        skill_id_from_draft(skill)
        for edit in edits
        for skill in edit.get("skills", [])
    ]

    result: dict[str, Any] = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "reasoning": raw["reasoning"],
        "edits": edits,
    }
    if "ranking_details" in raw:
        if not isinstance(raw["ranking_details"], Mapping):
            _fail("invalid_ranking_details", "$.ranking_details", "Expected an object.")
        result["ranking_details"] = dict(raw["ranking_details"])
    if "_evidence_sidecar" in raw:
        sidecar_raw = _require_mapping(raw["_evidence_sidecar"], "$._evidence_sidecar")
        unknown = sorted(set(sidecar_raw) - set(patch_skill_ids))
        if unknown:
            _fail(
                "unknown_evidence_sidecar_skill_id",
                "$._evidence_sidecar",
                f"Sidecar IDs do not occur in this patch: {unknown}.",
            )
        sidecar: dict[str, list[dict[str, str]]] = {}
        for skill_id in sorted(sidecar_raw):
            if not SKILL_ID_RE.fullmatch(skill_id):
                _fail("invalid_skill_id", f"$._evidence_sidecar.{skill_id}", "Expected a full SHA-256 skill ID.")
            refs_raw = _require_list(sidecar_raw[skill_id], f"$._evidence_sidecar.{skill_id}")
            sidecar[skill_id] = normalise_evidence(
                _validate_evidence_ref(
                    item,
                    path=f"$._evidence_sidecar.{skill_id}[{index}]",
                )
                for index, item in enumerate(refs_raw)
            )
        result["_evidence_sidecar"] = sidecar
    return result


def validate_model_patch(
    value: str | Mapping[str, Any],
    *,
    known_skill_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate untrusted model output; system metadata is always rejected."""

    return parse_patch_payload(
        value,
        known_skill_ids=known_skill_ids,
        allow_ranking_details=False,
        allow_trusted_metadata=False,
    )


def serialise_patch_payload(value: Mapping[str, Any]) -> str:
    return canonical_json_text(
        parse_patch_payload(value, allow_trusted_metadata=True)
    )


def _evidence_for_skill(
    patch: Mapping[str, Any],
    skill_id: str,
) -> list[dict[str, str]]:
    sidecar = patch.get("_evidence_sidecar", {})
    return list(sidecar.get(skill_id, [])) if isinstance(sidecar, Mapping) else []


def apply_structured_patch_with_report(
    library: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply an ID-addressed patch transactionally to the main library."""

    base = validate_skill_library(library)
    known_ids = {skill["skill_id"] for skill in base["skills"]}
    canonical_patch = parse_patch_payload(
        patch,
        known_skill_ids=known_ids,
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )
    records = [dict(skill) for skill in base["skills"]]
    reports: list[dict[str, Any]] = []

    def find_index(skill_id: str) -> int:
        for index, record in enumerate(records):
            if record["skill_id"] == skill_id:
                return index
        _fail(
            "target_removed_during_patch",
            "$.edits",
            f"Target {skill_id!r} is no longer present.",
        )

    def build_records(edit: Mapping[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for draft in edit.get("skills", []):
            skill_id = skill_id_from_draft(draft)
            output.append(build_atomic_skill_record(
                draft,
                _evidence_for_skill(canonical_patch, skill_id),
            ))
        return output

    def merge_existing(new_record: Mapping[str, Any]) -> tuple[bool, str]:
        for index, existing in enumerate(records):
            if existing["skill_id"] == new_record["skill_id"]:
                records[index] = merge_verified_evidence(
                    existing,
                    new_record["evidence"],
                )
                return True, "merged_existing"
        return False, "inserted"

    for edit_index, edit in enumerate(canonical_patch["edits"], start=1):
        op = edit["op"]
        target = edit.get("target_skill_id")
        report: dict[str, Any] = {
            "index": edit_index,
            "op": op,
            "target_skill_id": target,
        }
        if op == "append":
            statuses = []
            for new_record in build_records(edit):
                merged, status = merge_existing(new_record)
                if not merged:
                    records.append(new_record)
                statuses.append({"skill_id": new_record["skill_id"], "status": status})
            report.update(status="applied_append", skill_statuses=statuses)
        elif op == "insert_after":
            target_index = find_index(str(target))
            insert_at = target_index + 1
            statuses = []
            for new_record in build_records(edit):
                merged, status = merge_existing(new_record)
                if not merged:
                    records.insert(insert_at, new_record)
                    insert_at += 1
                statuses.append({"skill_id": new_record["skill_id"], "status": status})
            report.update(status="applied_insert_after", skill_statuses=statuses)
        elif op == "replace":
            target_index = find_index(str(target))
            records.pop(target_index)
            new_record = build_records(edit)[0]
            merged, status = merge_existing(new_record)
            if not merged:
                records.insert(target_index, new_record)
            report.update(
                status="applied_replace",
                skill_statuses=[
                    {"skill_id": new_record["skill_id"], "status": status}
                ],
            )
        else:
            target_index = find_index(str(target))
            records.pop(target_index)
            report["status"] = "applied_delete"
        reports.append(report)

    updated = validate_skill_library({
        "schema_version": SKILL_LIBRARY_SCHEMA_VERSION,
        "skills": records,
    })
    return updated, reports


def apply_structured_patch(
    library: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    updated, _ = apply_structured_patch_with_report(library, patch)
    return updated


def build_solver_projection(library: str | Mapping[str, Any]) -> str:
    """Render only active solver semantics as readable English text."""

    canonical = parse_skill_library(library)
    blocks: list[str] = []
    for index, skill in enumerate(canonical["skills"], start=1):
        blocks.append("\n".join((
            f"[Atomic Skill {index}]",
            f"Name: {skill['title']}",
            "Problem-solving guidance: "
            f"{skill['solver_guidance']}",
        )))
    return "\n\n".join(blocks)


def serialise_solver_projection(
    library: str | Mapping[str, Any],
    **_unused: Any,
) -> str:
    return build_solver_projection(library)


def write_json_atomic(path: str, value: Any) -> None:
    """Atomically persist generic JSON; callers validate their own schema."""

    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as file:
        file.write(pretty_json_text(value))
    os.replace(temporary, absolute)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as file:
        return strict_json_loads(file.read())


def write_skill_library_atomic(path: str, library: Mapping[str, Any]) -> str:
    canonical = validate_skill_library(library)
    write_json_atomic(path, canonical)
    return compute_library_hash(canonical)


def read_skill_library(path: str) -> dict[str, Any]:
    return validate_skill_library(read_json(path))


SKILL_DRAFT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(ROLE_FIELDS),
    "properties": {
        field: {"type": "string", "minLength": 1}
        for field in ROLE_FIELDS
    },
}

PATCH_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "reasoning", "edits"],
    "properties": {
        "schema_version": {"const": PATCH_SCHEMA_VERSION},
        "reasoning": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "skills"],
                        "properties": {
                            "op": {"const": "append"},
                            "skills": {
                                "type": "array",
                                "minItems": 1,
                                "items": SKILL_DRAFT_JSON_SCHEMA,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "target_skill_id", "skills"],
                        "properties": {
                            "op": {"const": "insert_after"},
                            "target_skill_id": {
                                "type": "string",
                                "pattern": r"^skill_sha256_[0-9a-f]{64}$",
                            },
                            "skills": {
                                "type": "array",
                                "minItems": 1,
                                "items": SKILL_DRAFT_JSON_SCHEMA,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "target_skill_id", "skills"],
                        "properties": {
                            "op": {"const": "replace"},
                            "target_skill_id": {
                                "type": "string",
                                "pattern": r"^skill_sha256_[0-9a-f]{64}$",
                            },
                            "skills": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "items": SKILL_DRAFT_JSON_SCHEMA,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "target_skill_id"],
                        "properties": {
                            "op": {"const": "delete"},
                            "target_skill_id": {
                                "type": "string",
                                "pattern": r"^skill_sha256_[0-9a-f]{64}$",
                            },
                        },
                    },
                ]
            },
        },
    },
}


__all__ = [
    "AtomicSkillDraft",
    "AtomicSkillRecord",
    "EvidenceRef",
    "PATCH_JSON_SCHEMA",
    "PATCH_OPS",
    "PATCH_SCHEMA_VERSION",
    "ROLE_FIELDS",
    "SKILL_DRAFT_JSON_SCHEMA",
    "SKILL_ID_RE",
    "SKILL_LIBRARY_SCHEMA_VERSION",
    "SOLVER_VIEW_SCHEMA_VERSION",
    "SkillJsonError",
    "SkillLibrary",
    "StructuredSkillError",
    "apply_structured_patch",
    "apply_structured_patch_with_report",
    "build_atomic_skill_record",
    "build_solver_projection",
    "canonical_json_bytes",
    "canonical_json_text",
    "compute_library_hash",
    "compute_skill_id",
    "dump_skill_library",
    "empty_skill_library",
    "make_active_skill_library",
    "merge_verified_evidence",
    "normalise_atomic_skill_draft",
    "normalise_evidence",
    "parse_atomic_skill_draft",
    "parse_atomic_skill_record",
    "parse_patch_edit",
    "parse_patch_payload",
    "parse_skill_library",
    "parse_skill_library_text",
    "payload_hash",
    "pretty_json_text",
    "read_json",
    "read_skill_library",
    "semantic_payload",
    "serialise_patch_payload",
    "serialise_skill_library",
    "serialise_solver_projection",
    "skill_behavior_hash",
    "skill_id_from_draft",
    "skill_library_to_dict",
    "strict_json_loads",
    "validate_model_patch",
    "validate_skill_draft",
    "validate_skill_library",
    "write_json_atomic",
    "write_skill_library_atomic",
]
