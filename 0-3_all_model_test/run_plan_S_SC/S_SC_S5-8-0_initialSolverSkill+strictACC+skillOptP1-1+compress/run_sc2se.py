"""Run SC2SE from frozen Flash answers and grade both final-library responses."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCORER = ROOT / "0-2-2_unified_evaluation"
DATA_ROOT = ROOT / "0-0_datasets"
DATASETS = json.loads((DATA_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))


def checked_inputs(dataset, questions, initial, limit):
    rows = json.loads(questions.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{dataset}: expected a nonempty list of questions")
    records = [json.loads(line) for line in initial.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"{dataset}: duplicate initial-answer IDs")
    provenance = initial.with_suffix(".version.json")
    if not provenance.is_file():
        raise ValueError(f"{dataset}: the matching initial-answer version file is required: {provenance}")
    state = json.loads(provenance.read_text(encoding="utf-8"))
    contract = state["contract"]
    spec = spec_from_file_location("shared_initials", ROOT / "0-1_initial_response/generate_initial_responses.py")
    generator = module_from_spec(spec)
    spec.loader.exec_module(generator)
    if (contract.get("dataset") != dataset or contract.get("endpoint") != generator.BASE_URL
            or contract.get("model") != generator.MODEL or contract.get("thinking") != "disabled"
            or contract.get("temperature") != 0):
        raise ValueError(f"{dataset}: expected initial answers from official DeepSeek Flash in non-thinking mode.")
    ids = []
    for i, question in enumerate(rows[:limit]):
        key = f"{dataset}_{i}"
        answer = by_id.get(key, {})
        prompt = next((question[k] for k in ("query", "problem", "q", "question", "prompt") if question.get(k)), None)
        if not prompt or answer.get("query") != prompt:
            raise ValueError(f"{key}: question or matching initial answer missing")
        if not isinstance(answer.get("response"), str) or not answer["response"].strip() or "pred" not in answer:
            raise ValueError(f"{key}: incomplete initial answer")
        record = state["records"].get(key, {})
        if record.get("row") != answer:
            raise ValueError(f"{key}: initial answer differs from its recorded version")
        generator.validate_record(record, key, prompt)
        ids.append(key)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable model calls, tool execution and final scoring.")
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Select a dataset (default: demo); repeat to combine.")
    parser.add_argument("--limit", type=int, help="Take the first N questions per selected dataset.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent questions per dataset.")
    parser.add_argument("--initial-dir", type=Path, default=ROOT / "runs/initial", help="Directory of matching deepseek-flash initial-answer JSONL files.")
    parser.add_argument("--output", type=Path, help="Create a new directory for this run.")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.dataset and len(set(args.dataset)) != len(args.dataset):
        parser.error("select each dataset once")

    todo = []
    for name in args.dataset or ["demo"]:
        questions = DATA_ROOT / DATASETS[name]["file"]
        initial = args.initial_dir / f"deepseek-flash_{name}_first_response.jsonl"
        for path in (questions, initial):
            if not path.is_file():
                parser.error(f"missing {path}; see the data preparation instructions in README.md")
        ids = checked_inputs(name, questions, initial, args.limit)
        print(f"{name}: {len(ids)} questions matched to shared initial answers.", flush=True)
        todo.append((name, questions, initial, ids))
    if not args.run:
        print("Input check complete. Add --run to execute and score.")
        return

    missing = [key for key in ("DEEPSEEK_API_KEY",) if not os.environ.get(key)]
    if missing:
        parser.error("set the following environment variables: " + ", ".join(missing))
    sys.path.insert(0, str(ROOT))
    from sc2se_runner import run_dataset

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or ROOT / "runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stages = {"initial": "initial_response", "corrected": "final_response"}
    output_paths = {stage: [] for stage in stages}
    print(f"Run directory: {out}", flush=True)
    for name, questions, initial, ids in todo:
        result = run_dataset(dataset_key=name, data_path=questions, initial_response_file=initial,
                             output_dir=out / name, ids=ids, workers=args.workers)
        rows = [json.loads(line) for line in result.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
            raise ValueError(f"{name}: final-library answers differ from the selected question IDs.")
        prompts = dict(zip(ids, [q["q"] for q in json.loads(questions.read_text(encoding="utf-8-sig"))]))
        for stage, field in stages.items():
            logs = []
            for row in rows:
                key = row["id"]
                if row["task_description"] != prompts[key].strip():
                    raise ValueError(f"{key}: final-library question differs from the selected question.")
                if not isinstance(row[field], str):
                    raise ValueError(f"{key}: missing {field} for final-library evaluation.")
                logs.append({"id": key, "query": prompts[key], "final_response": row[field]})
            path = out / name / f"responses_{stage}.json"
            path.write_text(json.dumps({"logs": logs}, ensure_ascii=False), encoding="utf-8")
            output_paths[stage].extend(["--dataset", name, str(questions), str(path)])

    for stage, groups in output_paths.items():
        inputs = out / f"evaluation_input_{stage}.json"
        scores = out / f"evaluation_{stage}.json"
        subprocess.run([sys.executable, str(SCORER / "prepare_inputs.py"), *groups, "--output", str(inputs)],
                       check=True)
        subprocess.run([sys.executable, str(SCORER / "evaluate.py"), "--run", "--input_json", str(inputs),
                        "--output_json", str(scores), "--cache_dir", str(out / f"evaluation_cache_{stage}")],
                       check=True)
        summary = json.loads(scores.read_text(encoding="utf-8"))
        if (summary.get("status") != "completed" or summary.get("acc") is None
                or summary["n_completed"] != sum(len(entry[3]) for entry in todo)):
            raise RuntimeError(f"Evaluation incomplete; inspect {scores}")
        print(f"{stage}: {summary['n_correct']}/{summary['n_completed']} ({summary['acc']:.2f}%). {scores}")


if __name__ == "__main__":
    main()
