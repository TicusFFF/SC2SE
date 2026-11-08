"""Exercise the same inference-to-scoring entry on demo and benchmark inputs."""
import importlib.util
import hashlib
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = next((ROOT / "0-3_all_model_test").rglob("run_sc2se.py"))
sys.path.insert(0, str(ROOT))
import sc2se_runner


class ReleaseEntryTest(unittest.TestCase):
    def test_optimizer_probe_retries_a_malformed_reply(self):
        module_path = str(sc2se_runner.METHOD / "mod_B0_SkillOpt")
        with patch.object(sys, "path", [module_path, str(sc2se_runner.METHOD), *sys.path]):
            from skillopt.engine import trainer

        probe = trainer.ReflACTTrainer({"optimizer_probe": True}, None)._probe_optimizer
        good = ('{"status":"ok"}', {})
        malformed = ('{"status\n":"ok"}', {})
        with patch.object(trainer, "chat_optimizer", side_effect=[good]) as chat:
            probe()
            chat.assert_called_once()
        with patch.object(trainer, "chat_optimizer", side_effect=[malformed, good]) as chat:
            probe()
            first, retry = chat.call_args_list
            self.assertEqual(first.kwargs["user"], '{"status":"ok"}')
            self.assertNotEqual(first.kwargs["user"], retry.kwargs["user"])
        with patch.object(trainer, "chat_optimizer", side_effect=[malformed, malformed]) as chat:
            with self.assertRaisesRegex(RuntimeError, "unexpected response"):
                probe()
            self.assertEqual(chat.call_count, 2)

    def test_native_training_receives_only_blind_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data, initial, out = base / "data.json", base / "initial.jsonl", base / "run"
            data.write_text(json.dumps([{"q": "Test question", "answer": "private reference"}]), encoding="utf-8")
            row = {"id": "demo_0", "query": "Test question", "response": "Frozen answer", "pred": "0"}
            initial.write_text(json.dumps(row) + "\n", encoding="utf-8")

            def native(cmd, *, env, cwd, check):
                self.assertEqual(Path(cmd[2]), sc2se_runner.NATIVE)
                question_file = Path(cmd[cmd.index("--dataset_path") + 1])
                self.assertEqual(json.loads(question_file.read_text(encoding="utf-8")),
                                 [{"id": "demo_0", "q": "Test question"}])
                staged_dir = Path(cmd[cmd.index("--initial_response_dir") + 1])
                staged = next(staged_dir.glob("*_first_response.jsonl"))
                self.assertEqual(json.loads(staged.read_text(encoding="utf-8")), row)
                for option in ("train.num_epochs=4", "evaluation.strict_validation_metric=forward_pass_rate",
                               "evaluation.use_gate=false", "env.bootstrap_shared_initial_response=true",
                               "evaluation.eval_final_skill=true", "--defer_report_scoring"):
                    self.assertIn(option, cmd)
                self.assertNotIn(str(data), cmd)
                report = {"evaluated_skill_version": 4, "skill_update_performed": False, "n": 1}
                (out / "final_skill_evaluation.json").write_text(json.dumps(report), encoding="utf-8")

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline"}), \
                 patch.object(sc2se_runner, "_local_tool", return_value=contextlib.nullcontext((4200,))), \
                 patch.object(sc2se_runner.subprocess, "run", side_effect=native) as train:
                result = sc2se_runner.run_dataset(dataset_key="demo", data_path=data, initial_response_file=initial,
                                                  output_dir=out, ids=["demo_0"])
                train.assert_called_once()
            self.assertEqual(result, out / "final_skill_evaluation/results.jsonl")

    def test_shared_dataset_entry(self):
        spec = importlib.util.spec_from_file_location("flash_entry", ENTRY)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for dataset in ("demo", "hmmt_feb_2025"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as work:
                base = Path(work)
                out = base / "run"
                data_root = entry.DATA_ROOT if dataset == "demo" else base / "datasets"
                questions = data_root / entry.DATASETS[dataset]["file"]
                if dataset != "demo":
                    questions.parent.mkdir(parents=True)
                    questions.write_text(json.dumps([{"q": "What is six times seven?", "answer": "42"}]), encoding="utf-8")
                row = json.loads(questions.read_text(encoding="utf-8"))[0]
                query = row["q"]
                sample_id = f"{dataset}_0"
                fixture = {"id": sample_id, "query": query, "response": "fixture", "pred": "",
                           "finish_reason": "length"}
                (base / f"deepseek-flash_{dataset}_first_response.jsonl").write_text(json.dumps(fixture)+"\n", encoding="utf-8")
                def digest(text):
                    return hashlib.sha256(text.encode("utf-8")).hexdigest()

                provenance = base / f"deepseek-flash_{dataset}_first_response.version.json"
                call = {"requested_model": "deepseek-flash", "returned_model": "deepseek-flash",
                        "prompt_sha256": digest(query), "response_sha256": digest(fixture["response"])}
                state = {"contract": {"model": "deepseek-flash", "temperature": 0,
                                      "dataset": dataset, "endpoint": "https://api.deepseek.com",
                                      "thinking": "disabled"},
                         "records": {sample_id: {"row": fixture, "calls": [call], "prediction_sha256": digest("")}}}
                provenance.write_text(json.dumps(state), encoding="utf-8")

                def verifier_fixture(*, output_dir, dataset_key, ids, **kwargs):
                    self.assertEqual(dataset_key, dataset)
                    self.assertEqual(ids, [sample_id])
                    directory = output_dir / "final_skill_evaluation"
                    directory.mkdir(parents=True)
                    path = directory / "results.jsonl"
                    result = {"id": sample_id, "task_description": query.strip(),
                              "initial_response": "Final answer: 987654321",
                              "final_response": f"Final answer: {row['answer']}"}
                    path.write_text(json.dumps(result) + "\n", encoding="utf-8")
                    return path

                argv = [str(ENTRY), "--dataset", dataset, "--limit", "1", "--initial-dir", str(base),
                        "--output", str(out), "--run"]
                env = {"DEEPSEEK_API_KEY": "offline", "HTTPS_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}
                with patch.dict(os.environ, env), patch.object(sys, "argv", argv), patch.object(entry, "DATA_ROOT", data_root):
                    with patch.object(sc2se_runner, "run_dataset", side_effect=verifier_fixture) as run:
                        entry.main()
                        run.assert_called_once()
                first = json.loads((out / "evaluation_initial.json").read_text(encoding="utf-8"))
                self.assertEqual((first["n_correct"], first["n_completed"]), (0, 1))
                scores = json.loads((out / "evaluation_corrected.json").read_text(encoding="utf-8"))
                self.assertEqual([r["id"] for r in scores["rows"]], [sample_id])
                self.assertEqual((scores["acc"], scores["n_completed"]), (100.0, 1))
                state["contract"]["model"] = "different-model"
                provenance.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "expected initial answers from official DeepSeek Flash"):
                    entry.checked_inputs(dataset, questions,
                                         base / f"deepseek-flash_{dataset}_first_response.jsonl", 1)


if __name__ == "__main__":
    unittest.main()
