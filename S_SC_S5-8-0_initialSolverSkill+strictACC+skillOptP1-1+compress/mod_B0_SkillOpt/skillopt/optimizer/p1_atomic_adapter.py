"""Adapt S5 solver-skill JSON to P1 atomic editing.

P1 owns proposal application and fallback rescue.
S5 owns the two-field solver-role contract and its public skill format.

An edit proposal is the outer ranking/budget unit.
Each skill in its `skills` array is an independent solver rule.
"""
from __future__ import annotations

from pathlib import Path

from skillopt.evaluation.gate import GateResult
from skillopt.optimizer import atomic_document as p1
from skillopt.optimizer.structured_skills import (
    ROLE_FIELDS,
    build_atomic_skill_record,
    build_solver_projection,
    canonical_json_text,
    empty_skill_library,
    make_active_skill_library,
    normalise_evidence,
    parse_patch_payload,
    parse_skill_library_text,
    serialise_skill_library,
    skill_id_from_draft,
    strict_json_loads,
    validate_skill_draft,
    write_json_atomic,
)


def from_markdown(text: str) -> str:
    if str(text).strip():
        raise ValueError(
            "S5 requires a structured two-field skill library, "
            "not an unstructured Markdown seed."
        )
    return serialise_skill_library(empty_skill_library())


def render_target(document: str) -> str:
    return serialise_skill_library(parse_skill_library_text(document))


def render_optimizer(document: str) -> str:
    return render_target(document)


def load_skill(path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return render_target(text)


def save_skill(path, document: str) -> None:
    path = Path(path)
    library = parse_skill_library_text(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(path.with_suffix(".json")), library)
    path.with_suffix(".md").write_text(
        build_solver_projection(library),
        encoding="utf-8",
    )


def with_text_update(document, function, *args, **kwargs):
    raise ValueError(
        "Unscreened text updates are disabled for S5 solver skills. "
        "Use the role-screened P1 edit-proposal path."
    )


def _payload(draft: dict) -> dict:
    draft = validate_skill_draft(draft)
    return {
        "title": draft["title"],
        "content": canonical_json_text(draft),
    }


def _to_p1(document: str) -> tuple[dict, dict[str, str]]:
    used = set()
    segments = []
    targets = {}

    for record in parse_skill_library_text(document)["skills"]:
        draft = {field: record[field] for field in ROLE_FIELDS}
        segment = p1._payload_segment(_payload(draft), used)
        segments.append(segment)
        targets[record["skill_id"]] = segment["skill_id"]

    return {
        "version": p1.VERSION,
        "segments": segments,
    }, targets


def _patch_to_p1(patch: dict, targets: dict[str, str]) -> dict:
    patch = parse_patch_payload(
        patch,
        known_skill_ids=set(targets),
        allow_ranking_details=True,
        allow_trusted_metadata=True,
    )

    edits = []
    for item in patch["edits"]:
        target = item.get("target_skill_id")
        edits.append({
            "op": item["op"],
            "target": targets[target] if target else "",
            "skills": [
                _payload(draft)
                for draft in item.get("skills", [])
            ],
        })

    return {
        "reasoning": patch["reasoning"],
        "edits": edits,
    }


def _evidence_pool(documents: list[str], patch: dict) -> dict:
    by_id = {}

    for document in documents:
        for record in parse_skill_library_text(document)["skills"]:
            skill_id = record["skill_id"]
            by_id[skill_id] = normalise_evidence([
                *by_id.get(skill_id, []),
                *record["evidence"],
            ])

    for skill_id, references in patch.get(
        "_evidence_sidecar", {}
    ).items():
        by_id[skill_id] = normalise_evidence([
            *by_id.get(skill_id, []),
            *references,
        ])

    return by_id


def _from_p1(document: dict, evidence: dict) -> str:
    records = []
    seen = set()

    for segment in p1._validate(document)["segments"]:
        if segment["kind"] != "skill":
            if segment["content"].strip():
                raise p1.AtomicDocumentError(
                    "S5 solver libraries cannot contain "
                    "unreviewed text/protected segments."
                )
            continue

        prefix = "\n\n### " + segment["title"].strip() + "\n"
        content = segment["content"]
        if not content.startswith(prefix):
            raise p1.AtomicDocumentError(
                "Unexpected P1 skill-segment representation."
            )

        draft = validate_skill_draft(
            strict_json_loads(content[len(prefix):].strip())
        )
        if draft["title"] != segment["title"]:
            raise p1.AtomicDocumentError(
                "The P1 segment title differs from the S5 skill title."
            )

        skill_id = skill_id_from_draft(draft)

        # Preserve S5's existing semantic-ID uniqueness constraint.
        # This is exact two-field identity, not semantic LLM deduplication.
        if skill_id in seen:
            continue

        seen.add(skill_id)
        records.append(
            build_atomic_skill_record(
                draft,
                evidence.get(skill_id, []),
            )
        )

    return serialise_skill_library(
        make_active_skill_library(records)
    )


def apply_patch_with_report(
    document: str,
    patch: dict,
) -> tuple[str, list]:
    base, targets = _to_p1(document)
    candidate, report = p1.apply_patch_with_report(
        base,
        _patch_to_p1(patch, targets),
    )
    evidence = _evidence_pool([document], patch)
    return _from_p1(candidate, evidence), report


def rescue_update(
    base_doc: str,
    ranked_patch: dict,
    current_score: float,
    best_doc: str,
    best_score: float,
    best_step: int,
    global_step: int,
    evaluate,
    out_dir,
    metric: str = "hard",
    mixed_weight: float = 0.5,
) -> tuple[GateResult, dict]:
    base, targets = _to_p1(base_doc)
    best, _ = _to_p1(best_doc)
    evidence = _evidence_pool(
        [base_doc, best_doc],
        ranked_patch,
    )

    def evaluate_s5(document, trial_dir):
        return evaluate(
            _from_p1(document, evidence),
            trial_dir,
        )

    gate, report = p1.rescue_update(
        base_doc=base,
        ranked_patch=_patch_to_p1(ranked_patch, targets),
        current_score=current_score,
        best_doc=best,
        best_score=best_score,
        best_step=best_step,
        global_step=global_step,
        evaluate=evaluate_s5,
        out_dir=out_dir,
        metric=metric,
        mixed_weight=mixed_weight,
    )

    return GateResult(
        action=gate.action,
        current_skill=_from_p1(gate.current_skill, evidence),
        current_score=gate.current_score,
        best_skill=_from_p1(gate.best_skill, evidence),
        best_score=gate.best_score,
        best_step=gate.best_step,
    ), report
