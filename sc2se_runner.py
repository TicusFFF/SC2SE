"""Run the four native SC2SE updates with shared initial answers and a private Python tool."""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
METHOD = ROOT / "S_SC_S5-8-0_initialSolverSkill+strictACC+skillOptP1-1+compress"
NATIVE = METHOD / "S5_bridge/train_S5.py"
PROFILE = ROOT / "0-4_API/0_curr_API"
REGISTRY = ROOT / "0-4_API/2_model_capacity/model_capacities.json"
ALIAS = "dso_DSv41Flash"


def _serve_tool(work):
    from gevent import sleep
    from gevent.pywsgi import WSGIServer

    os.environ["CODE_INTERPRETER_WORK_DIR"] = str(work)
    os.environ["SPQ_PYTHON_TOOL_LOCK_DIR"] = str(work / "locks")
    (work / "upload_file").mkdir()
    sys.path.insert(0, str(METHOD / "mod_S3_Forward/repro/scripts"))
    import pack_api_windows as tool

    server = WSGIServer(("127.0.0.1", 0), tool.app, log=None)
    server.start()
    ready = work / "port"
    pending = ready.with_suffix(".tmp")
    pending.write_text(str(server.server_port), encoding="ascii")
    pending.replace(ready)
    try:
        while work.exists() and not (work / "stop").exists():
            sleep(0.2)
    finally:
        server.stop(timeout=5)
        tool.ci.stop_kernel(os.getpid())


@contextlib.contextmanager
def _local_tool(work):
    # Generated Python runs in this separate process without the model API keys.
    allowed = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "LANG",
        "LC_ALL", "CONDA_PREFIX", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLCONFIGDIR=str(work / "matplotlib"))
    proc = subprocess.Popen(
        [sys.executable, "-B", str(Path(__file__).resolve()), "_tool", str(work)],
        env=env, cwd=work,
    )
    try:
        deadline = time.monotonic() + 60
        ready = work / "port"
        while not ready.exists():
            if proc.poll() is not None:
                raise RuntimeError(f"Python tool exited with status {proc.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Python tool did not start within 60 seconds.")
            time.sleep(0.1)
        yield (int(ready.read_text(encoding="ascii")),)
    finally:
        (work / "stop").touch()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)


def _routed_registry(work, endpoint):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = registry["profiles"]["deepseek_official"][ALIAS]
    if endpoint == entry["identity"]["base_url"]:
        return REGISTRY
    # The local test uses the same capacities and tokenizer as generation.
    entry["identity"]["base_url"] = endpoint
    counter = entry["token_counting"]
    for field in ("tokenizer_name_or_path", "encoding_module_path"):
        counter[field] = str((REGISTRY.parent / counter[field]).resolve())
    counter["paths_relative_to_registry"] = False
    destination = work / "registry.json"
    destination.write_text(json.dumps(registry), encoding="utf-8")
    return destination


# Flash settings from the four-round paper launcher.
FLASH_OPTIONS = {'train.num_epochs': 4,
 'train.train_size': 0,
 'train.rollout_size': 0,
 'train.accumulation': 1,
 'train.seed': 42,
 'gradient.minibatch_size': 8,
 'gradient.merge_batch_size': 8,
 'gradient.failure_only': False,
 'gradient.max_analyst_rounds': 3,
 'optimizer.skill_update_mode': 'patch',
 'optimizer.learning_rate': 4,
 'optimizer.min_learning_rate': 1,
 'optimizer.lr_scheduler': 'cosine',
 'optimizer.lr_control_mode': 'fixed',
 'optimizer.use_slow_update': False,
 'optimizer.slow_update_samples': 20,
 'optimizer.slow_update_gate_with_selection': False,
 'optimizer.longitudinal_pair_policy': 'mixed',
 'optimizer.use_meta_skill': False,
 'model.short_max_tokens': 8192,
 'model.default_max_tokens': 8192,
 'model.rewrite_reasoning_effort': '',
 'model.rewrite_max_completion_tokens': 8192,
 'model.contrastive_soft_max_tokens': 8192,
 'compression.trigger_tokens': 131072,
 'evaluation.use_gate': False,
 'evaluation.strict_validation_metric': 'forward_pass_rate',
 'evaluation.gate_metric': 'hard',
 'evaluation.gate_mixed_weight': 0.5,
 'evaluation.sel_env_num': 0,
 'evaluation.test_env_num': 0,
 'evaluation.eval_test': False,
 'evaluation.eval_final_skill': True,
 'evaluation.use_contrastive_soft': False,
 'evaluation.contrastive_soft_sample_limit': 0,
 'evaluation.contrastive_soft_fail_fast': False,
 'env.use_initial_skill': False,
 'env.bootstrap_shared_initial_response': True,
 'env.optimizer_probe': True}


def run_dataset(*, dataset_key, data_path, initial_response_file, output_dir, ids, workers=1):
    """Only questions and the selected shared responses are given to the learning process."""
    from importlib.util import module_from_spec, spec_from_file_location

    profile_spec = spec_from_file_location("sc2se_generation_profile", PROFILE / "api_profiles.py")
    profile = module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile)
    config = profile.get_api_config("deepseek_official")
    initial_rows = {r["id"]: r for r in (
        json.loads(line) for line in Path(initial_response_file).read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    dataset_rows = json.loads(Path(data_path).read_text(encoding="utf-8-sig"))
    pool = [{"id": key, "q": row["q"]} for key, row in zip(ids, dataset_rows)]
    if (workers < 1 or not pool or len(pool) != len(ids)
            or any(row["id"] != f"{dataset_key}_{i}" or initial_rows[row["id"]]["query"] != row["q"]
                   for i, row in enumerate(pool))):
        raise ValueError("Question IDs or prompts differ from their shared initial answers.")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(FLASH_OPTIONS)
    cfg.update({"env.limit": 0, "env.rollout_workers": workers, "gradient.analyst_workers": workers,
                "evaluation.contrastive_soft_workers": workers})
    options = [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in cfg.items()]
    with tempfile.TemporaryDirectory(prefix="sc2se-") as tmp:
        work = Path(tmp)
        questions = work / "questions.json"
        questions.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
        bootstrap = work / f"{profile.model_file_tag(ALIAS)}_{dataset_key}_first_response.jsonl"
        with bootstrap.open("x", encoding="utf-8") as file:
            for key in ids:
                row = initial_rows[key]
                file.write(json.dumps({field: row[field] for field in ("id", "query", "response", "pred")},
                                      ensure_ascii=False) + "\n")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PROFILE), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                   SC2SE_RESPONSE_METADATA=str(out / "calls.models.jsonl"),
                   SPQ_API_PROFILE="deepseek_official", SPQ_PYTHON_TOOL_LOCK_DIR=str(work / "locks"),
                   SPQ_COMPRESSION_CAPACITIES_PATH=str(_routed_registry(work, config["base_url"])))
        with _local_tool(work) as ports:
            cmd = [
                sys.executable, "-B", str(NATIVE),
                "--config", "mod_B0_SkillOpt/configs/forward_math/default.yaml",
                "--dataset_path", str(questions), "--dataset_key", dataset_key,
                "--initial_response_dir", str(work), "--out_root", str(out), "--defer_report_scoring",
                "--forward_model", ALIAS, "--a0_model", ALIAS, "--answer_extractor_model", ALIAS,
                "--branch_math_model", ALIAS, "--max_cur_turn", "3", "--forward_temperature", "0.0",
                "--a0_temperature", "0.0", "--python_tool_ports", ",".join(map(str, ports)),
                "--no_start_python_tool_server", "--reuse_existing_python_tool_servers",
                "--skip_python_tool_dependency_setup", "--python_tool_execution_timeout_seconds", "120",
                "--python_tool_http_max_attempts", "4", "--python_tool_code_soft_warning_calls", "50",
                "--python_tool_code_hard_stop_calls", "70", "--python_tool_duplicate_soft_warning_calls", "10",
                "--python_tool_duplicate_hard_block_calls", "20", "--agent2_execute_retries", "10",
                "--agent7_rewrite_retries", "10", "--agent_retry_wait_seconds", "10",
                "--llm_timeout_seconds", "960", "--llm_max_retries", "5", "--llm_stop_max_retries", "4",
                "--cfg-options", *options,
            ]
            subprocess.run(cmd, env=env, cwd=METHOD, check=True)
    report = json.loads((out / "final_skill_evaluation.json").read_text(encoding="utf-8"))
    if (report["evaluated_skill_version"] != 4 or report["skill_update_performed"]
            or report["n"] != len(ids)):
        raise ValueError(f"{dataset_key}: expected evaluation of the frozen fourth-round library.")
    return out / "final_skill_evaluation/results.jsonl"


if __name__ == "__main__" and sys.argv[1:2] == ["_tool"]:
    _serve_tool(Path(sys.argv[2]).resolve())
