"""Minimal atomic JSON document and original-gate fallback rescue."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from skillopt.evaluation.gate import GateResult, evaluate_gate

VERSION = 1
PAIRS = {
    "<!-- SLOW_UPDATE_START -->": "<!-- SLOW_UPDATE_END -->",
    "<!-- APPENDIX_START -->": "<!-- APPENDIX_END -->",
}
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}\s+")
LIST_ITEM = re.compile(r"^(?:[-+*]|\d+[.)])\s+")


class AtomicDocumentError(ValueError):
    """Invalid document or atomic operation; not an infrastructure error."""


def _validate(doc):
    if not isinstance(doc, dict) or set(doc) != {"version", "segments"}:
        raise AtomicDocumentError("document requires version and segments")
    if doc["version"] != VERSION or not isinstance(doc["segments"], list):
        raise AtomicDocumentError("unsupported document version or segments")
    used = set()
    for segment in doc["segments"]:
        if not isinstance(segment, dict):
            raise AtomicDocumentError("segment must be an object")
        kind = segment.get("kind")
        if not isinstance(kind, str):
            raise AtomicDocumentError("segment kind must be a string")
        fields = {"kind", "content"}
        if kind == "skill":
            fields |= {"skill_id", "title"}
        elif kind not in {"text", "protected"}:
            raise AtomicDocumentError("unknown segment kind")
        if set(segment) != fields or not isinstance(segment["content"], str):
            raise AtomicDocumentError("invalid segment fields")
        if kind == "skill":
            sid = segment["skill_id"]
            if (
                not isinstance(sid, str)
                or not re.fullmatch(r"s_[0-9a-f]{12}", sid)
                or sid in used
                or not isinstance(segment["title"], str)
            ):
                raise AtomicDocumentError("invalid or duplicate skill ID")
            used.add(sid)
    return copy.deepcopy(doc)


def _new_id(content, used):
    nonce = 0
    while True:
        value = hashlib.sha256(f"{content}\0{nonce}".encode("utf-8")).hexdigest()
        sid = "s_" + value[:12]
        if sid not in used:
            used.add(sid)
            return sid
        nonce += 1


def _fence_after(line, current):
    match = FENCE.match(line)
    if not match:
        return current
    marker, rest = match.groups()
    if current is None:
        return marker[0], len(marker)
    if marker[0] == current[0] and len(marker) >= current[1] and not rest.strip():
        return None
    return current


def _regions(text):
    """Split protected fields, ignoring marker-looking text in fenced code."""
    lines, result, normal = text.splitlines(keepends=True), [], []
    index, fence = 0, None
    while index < len(lines):
        line = lines[index]
        start = next(
            (marker for marker in PAIRS if line.lstrip().startswith(marker)),
            None,
        ) if fence is None else None
        if start:
            if normal:
                result.append(("text", "".join(normal)))
                normal = []
            block = []
            while index < len(lines):
                block.append(lines[index])
                index += 1
                if PAIRS[start] in block[-1]:
                    break
            else:
                raise AtomicDocumentError(f"unterminated protected field: {start}")
            result.append(("protected", "".join(block)))
            continue
        normal.append(line)
        fence = _fence_after(line, fence)
        index += 1
    if normal:
        result.append(("text", "".join(normal)))
    return result


def from_markdown(text):
    if not isinstance(text, str):
        raise AtomicDocumentError("Markdown must be a string")
    segments, used = [], set()

    def add(kind, content):
        if not content:
            return
        if kind == "skill":
            first = content.strip().splitlines()[0]
            segments.append({
                "kind": "skill",
                "skill_id": _new_id(content, used),
                "title": re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", first)[:100],
                "content": content,
            })
        elif segments and kind == "text" and segments[-1]["kind"] == "text":
            segments[-1]["content"] += content
        else:
            segments.append({"kind": kind, "content": content})

    for kind, content in _regions(text):
        if kind == "protected":
            add(kind, content)
            continue
        buffer, fence = [], None

        def flush():
            if buffer:
                add("skill", "".join(buffer))
                buffer.clear()

        for line in content.splitlines(keepends=True):
            if fence is not None:
                buffer.append(line)
                fence = _fence_after(line, fence)
                continue
            if FENCE.match(line):
                buffer.append(line)
                fence = _fence_after(line, None)
            elif not line.strip() or HEADING.match(line):
                flush()
                add("text", line)
            else:
                if LIST_ITEM.match(line):
                    flush()
                buffer.append(line)
        flush()
    return _validate({"version": VERSION, "segments": segments})


def render_target(doc):
    return "".join(segment["content"] for segment in _validate(doc)["segments"])


def render_optimizer(doc):
    parts = []
    for segment in _validate(doc)["segments"]:
        if segment["kind"] == "skill":
            parts.append(f"\n<!-- skill_id={segment['skill_id']} -->\n")
        parts.append(segment["content"])
    return "".join(parts)


def skill_len(doc):
    return len(render_target(doc))


def skill_hash(doc):
    return hashlib.sha256(render_target(doc).encode("utf-8")).hexdigest()[:16]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AtomicDocumentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise AtomicDocumentError(f"non-finite JSON number: {value}")


def load_skill(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return from_markdown(text)
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant,
        )
    except json.JSONDecodeError as exc:
        raise AtomicDocumentError(f"invalid skill JSON: {exc}") from exc
    return _validate(value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_skill(path, doc):
    path, doc = Path(path), _validate(doc)
    _write_json(path.with_suffix(".json"), doc)
    path.with_suffix(".md").write_text(render_target(doc), encoding="utf-8")


def with_text_update(doc, fn, *args, **kwargs):
    """Adapt original protected-field helpers while shielding atomic bodies."""
    old = _validate(doc)
    before = render_target(old)
    body = [s for s in old["segments"] if s["kind"] != "protected"]
    prefix = "".join(s["content"] for s in body).rstrip()
    if not before.startswith(prefix):
        raise AtomicDocumentError("protected fields must follow the atomic body")
    original_tail = _regions(before[len(prefix):])
    if any(kind == "text" and text.strip() for kind, text in original_tail):
        raise AtomicDocumentError("non-protected text follows the atomic body")
    marker = "P1_ATOMIC_BODY_SENTINEL"
    masked = marker + before[len(prefix):]
    after = fn(masked, *args, **kwargs)
    if not isinstance(after, str) or not after.startswith(marker):
        raise AtomicDocumentError("text update changed the protected body marker")
    tail = _regions(after[len(marker):])
    if any(kind == "text" and text.strip() for kind, text in tail):
        raise AtomicDocumentError("text update inserted non-protected body text")
    retained, remaining = [], len(prefix)
    for segment in body:
        count = min(remaining, len(segment["content"]))
        if count:
            retained.append({**segment, "content": segment["content"][:count]})
            remaining -= count
    retained.extend({"kind": kind, "content": text} for kind, text in tail)
    return _validate({"version": VERSION, "segments": retained})


def _payload_segment(payload, used, sid=None):
    # Merge output may repeat proposal metadata on a skill. Ignore these
    # known fields; do not treat them as per-skill evidence or store them.
    required = {"title", "content"}
    allowed = required | {"support_count", "source_type"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or not set(payload).issubset(allowed)
    ):
        raise AtomicDocumentError("skill payload requires title and content")
    title, content = payload["title"], payload["content"]
    if not isinstance(title, str) or not isinstance(content, str):
        raise AtomicDocumentError("skill title/content must be strings")
    if not title.strip() or not content.strip():
        raise AtomicDocumentError("skill title/content must not be empty")
    if any(marker in title + content for pair in PAIRS.items() for marker in pair):
        raise AtomicDocumentError("ordinary skill cannot contain protected markers")
    rendered = f"\n\n### {title.strip()}\n{content.strip()}\n"
    return {
        "kind": "skill", "skill_id": sid or _new_id(rendered, used),
        "title": title.strip(), "content": rendered,
    }


def expand_patch(doc, patch):
    """Expand selected proposals without imposing a second atomic budget."""
    doc = _validate(doc)
    used = {s["skill_id"] for s in doc["segments"] if s["kind"] == "skill"}
    edits = patch.get("edits") if isinstance(patch, dict) else None
    if not isinstance(edits, list):
        return [{"error": "patch.edits must be a list", "source_edit": 0}]
    atoms = []
    for index, edit in enumerate(edits, 1):
        try:
            if not isinstance(edit, dict):
                raise AtomicDocumentError("edit must be an object")
            op, target = edit.get("op"), edit.get("target", "")
            skills = edit.get("skills")
            if not isinstance(op, str) or op not in {
                "append", "insert_after", "replace", "delete"
            }:
                raise AtomicDocumentError("unknown operation")
            if not isinstance(target, str) or (op == "append" and target):
                raise AtomicDocumentError("invalid target")
            if op != "append" and not target:
                raise AtomicDocumentError("operation requires a skill ID")
            if not isinstance(skills, list):
                raise AtomicDocumentError("edit.skills must be a list")
            if op == "delete":
                if skills:
                    raise AtomicDocumentError("delete requires empty skills")
                atoms.append({"op": op, "target": target, "source_edit": index})
                continue
            if not skills or (op == "replace" and len(skills) != 1):
                raise AtomicDocumentError("replace requires one skill")
            proposal, anchors = [], [target]
            for payload in skills:
                segment = _payload_segment(
                    payload, used, sid=target if op == "replace" else None,
                )
                atom = {
                    "op": op, "target": target, "segment": segment,
                    "source_edit": index,
                }
                if op == "insert_after":
                    atom["anchors"] = list(anchors)
                    anchors.append(segment["skill_id"])
                proposal.append(atom)
            atoms.extend(proposal)
        except AtomicDocumentError as exc:
            atoms.append({"error": str(exc), "source_edit": index})
    return atoms


def _apply_atom(doc, atom):
    if "error" in atom:
        raise AtomicDocumentError(atom["error"])
    result = _validate(doc)
    segments, op = result["segments"], atom["op"]
    positions = {
        s["skill_id"]: i for i, s in enumerate(segments) if s["kind"] == "skill"
    }
    target = atom.get("target", "")
    if op == "append":
        index = next(
            (i for i, s in enumerate(segments) if s["kind"] == "protected"),
            len(segments),
        )
        segments.insert(index, copy.deepcopy(atom["segment"]))
    elif op == "insert_after":
        anchor = next(
            (sid for sid in reversed(atom["anchors"]) if sid in positions), None,
        )
        if anchor is None:
            raise AtomicDocumentError("insert_after target was not found")
        segments.insert(positions[anchor] + 1, copy.deepcopy(atom["segment"]))
    elif target not in positions:
        raise AtomicDocumentError(f"{op} target was not found")
    elif op == "replace":
        segments[positions[target]] = copy.deepcopy(atom["segment"])
    elif op == "delete":
        del segments[positions[target]]
    else:
        raise AtomicDocumentError("unknown operation")
    return _validate(result)


def apply_patch_with_report(doc, patch):
    current, reports = _validate(doc), []
    for index, atom in enumerate(expand_patch(current, patch), 1):
        row = {
            "index": index, "source_edit": atom["source_edit"],
            "op": atom.get("op"), "target": atom.get("target", ""),
        }
        try:
            candidate = _apply_atom(current, atom)
            changed = skill_hash(candidate) != skill_hash(current)
            current = candidate
            row["status"] = "applied" if changed else "skipped_no_change"
        except AtomicDocumentError as exc:
            row.update(status="skipped_invalid", reason=str(exc))
        reports.append(row)
    return current, reports


def rescue_update(
    base_doc, ranked_patch, current_score, best_doc, best_score, best_step,
    global_step, evaluate, out_dir, metric="hard", mixed_weight=0.5,
):
    """After package rejection, greedily rescue atoms from the old library."""
    current, best = _validate(base_doc), _validate(best_doc)
    start_score, start_best = current_score, best_score
    root, trials, accepted = Path(out_dir), [], 0
    atoms = expand_patch(current, ranked_patch)
    for index, atom in enumerate(atoms, 1):
        trial_dir = root / f"trial_{index:03d}"
        row = {
            "index": index, "source_edit": atom["source_edit"],
            "op": atom.get("op"), "target": atom.get("target", ""),
            "before_score": current_score,
            "edit": {
                "op": atom.get("op"), "target": atom.get("target", ""),
                "content": atom.get("segment", {}).get("content", ""),
            },
        }
        try:
            candidate = _apply_atom(current, atom)
        except AtomicDocumentError as exc:
            row.update(action="skipped_invalid", reason=str(exc))
        else:
            row["candidate_hash"] = skill_hash(candidate)
            if row["candidate_hash"] == skill_hash(current):
                row["action"] = "skipped_no_change"
            else:
                save_skill(trial_dir / "candidate_skill.json", candidate)
                # The caller owns score caching and complete selection rollout.
                # S5: incomplete Forward validation skips only this atom;
                # keep the accepted library/scores, never invent a gate score.
                # Other errors deliberately propagate.
                try:
                    hard, soft = evaluate(candidate, str(trial_dir))
                except RuntimeError as exc:
                    if str(exc) != (
                        "Forward-pass-rate selection must contain "
                        "exactly one complete Forward step."
                    ):
                        raise
                    row.update(action="skipped_incomplete_validation", reason=str(exc))
                    _write_json(trial_dir / "trial_result.json", row)
                    trials.append(row)
                    print(
                        f"    [fallback rescue] atom {index}: "
                        f"skipped_incomplete_validation; {exc}", flush=True,
                    )
                    continue
                gate = evaluate_gate(
                    candidate_skill=candidate, cand_hard=hard, cand_soft=soft,
                    current_skill=current, current_score=current_score,
                    best_skill=best, best_score=best_score, best_step=best_step,
                    global_step=global_step, metric=metric,
                    mixed_weight=mixed_weight,
                )
                row.update(action=gate.action, hard=hard, soft=soft)
                current, current_score = gate.current_skill, gate.current_score
                best, best_score, best_step = (
                    gate.best_skill, gate.best_score, gate.best_step,
                )
                accepted += gate.action in {"accept", "accept_new_best"}
        _write_json(trial_dir / "trial_result.json", row)
        trials.append(row)
    action = "reject"
    if current_score > start_score:
        action = "accept_new_best" if best_score > start_best else "accept"
    result = GateResult(
        action=action, current_skill=current, current_score=current_score,
        best_skill=best, best_score=best_score, best_step=best_step,
    )
    report = {
        "action": action, "accepted_atoms": accepted, "trials": trials,
        "initial_score": start_score, "final_score": current_score,
        "best_score": best_score, "best_step": best_step,
    }
    save_skill(root / "final_skill.json", current)
    _write_json(root / "rescue_result.json", report)
    return result, report
