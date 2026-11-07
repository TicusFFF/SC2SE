from __future__ import annotations

import argparse
import datetime
import importlib
import importlib.util
import json
import os
import psutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
B0_ROOT = ROOT / "mod_B0_SkillOpt"
S3_ROOT = ROOT / "mod_S3_Forward"
S3_PROGCO_ROOT = S3_ROOT / "upstream" / "progco"
S3_MATH_ROOT = S3_PROGCO_ROOT / "MATH"


for path in (
    ROOT,
    B0_ROOT,
    S3_PROGCO_ROOT,
    S3_MATH_ROOT,
):
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(0, path_text)


os.environ.setdefault(
    "PROJECT_PATH",
    str(S3_PROGCO_ROOT),
)

PYTHON_TOOL_STARTUP_CONNECT_TIMEOUT_SECONDS = 0.2
PYTHON_TOOL_STARTUP_WAIT_TIMEOUT_SECONDS = 10
PYTHON_TOOL_STARTUP_POLL_SECONDS = 0.1
PYTHON_TOOL_STARTUP_MAX_ATTEMPTS = 4


from skillopt.config import (  # noqa: E402
    flatten_config,
    is_structured,
    load_config,
)
from skillopt.engine.trainer import (  # noqa: E402
    ReflACTTrainer,
)
from skillopt.envs.forward_math.adapter import (  # noqa: E402
    ForwardMathAdapter,
)

from mod_C0_compress import (  # noqa: E402
    configure_context_runtime,
)

from S5_bridge.forward_runner import (  # noqa: E402
    ForwardSkillRunner,
)
from S5_bridge.data_sanitizer import (  # noqa: E402
    load_json_or_jsonl,
)


class S5MemoryLogger:
    """
    Minimal in-memory logger compatible with ProgCoAgent.inference.

    Each Forward runner keeps only the active sample in memory. Formal
    evidence is returned via output["S5"].
    """

    def __init__(self, args):
        self.args = args
        self.logs = []

    def init_sample(
        self,
        sample_id,
        query,
    ):
        self.logs = [
            {
                "id": sample_id,
                "query": query,
                "turns": [],
            }
        ]

    def update_turn(
        self,
        cur_turn,
        verify_code,
        response,
        result,
        execute_result,
        execute_content,
        feedback,
        reflection,
        code_feedback,
        revised_response=None,
        revised_result=None,
        a16_decision=None,
        next_verify_source=None,
    ):
        if not self.logs:
            raise RuntimeError(
                "S5MemoryLogger.update_turn called before init_sample."
            )

        turn_log = {
            "cur_turn": cur_turn,
            "verify_code": verify_code,
            "response": response,
            "result": result,
            "execute_result": execute_result,
            "execute_content": execute_content,
            "feedback": feedback,
            "reflection": reflection,
            "code_feedback": code_feedback,
        }

        if revised_response is not None:
            turn_log["revised_response"] = revised_response

        if revised_result is not None:
            turn_log["revised_result"] = revised_result

        if a16_decision is not None:
            turn_log["a16_decision"] = a16_decision

        if next_verify_source is not None:
            turn_log["next_verify_source"] = next_verify_source

        self.logs[-1]["turns"].append(
            turn_log
        )

    def update_final_response(
        self,
        response,
        result,
    ):
        if not self.logs:
            return

        self.logs[-1]["final_response"] = response
        self.logs[-1]["final_result"] = result

    def update_insights(
        self,
        insights,
    ):
        if not self.logs:
            return

        self.logs[-1]["insights"] = insights

    def save_log(self):
        pass


def _resolve_path(
    path_text: str | os.PathLike,
    *,
    base_dirs: tuple[Path, ...],
    fallback_base: Path,
    prefer_existing: bool = True,
) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    if prefer_existing:
        for base_dir in base_dirs:
            candidate = base_dir / path

            if candidate.exists():
                return candidate.resolve()

    return (
        fallback_base
        / path
    ).resolve()


def _resolve_under_B0(
    path_text: str | os.PathLike,
) -> str:
    return str(
        _resolve_path(
            path_text,
            base_dirs=(
                Path.cwd(),
                ROOT,
                B0_ROOT,
            ),
            fallback_base=B0_ROOT,
            prefer_existing=True,
        )
    )


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_python_file(
    module_name: str,
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {module_name}: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {module_name} from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def _default_python_tool_script() -> Path:
    candidates = [
        (
            ROOT
            / "mod_S3_Forward"
            / "repro"
            / "scripts"
            / "math_hard_test_parallel_python_tool.py"
        ),
        (
            ROOT.parent
            / "S3-1_Forward"
            / "repro"
            / "scripts"
            / "math_hard_test_parallel_python_tool.py"
        ),
    ]

    for path in candidates:
        if path.is_file():
            return path

    return candidates[0]


def resolve_python_tool_script_path(args) -> Path:
    if args.python_tool_script:
        return _resolve_path(
            args.python_tool_script,
            base_dirs=(
                Path.cwd(),
                ROOT,
                S3_ROOT,
                ROOT.parent,
            ),
            fallback_base=Path.cwd(),
            prefer_existing=True,
        )

    return _default_python_tool_script()


def is_port_open(port, startup_stats=None):
    if startup_stats is not None:
        startup_stats["probe_attempts"] += 1
    try:
        with socket.create_connection(
            ("127.0.0.1", port),
            timeout=PYTHON_TOOL_STARTUP_CONNECT_TIMEOUT_SECONDS,
        ):
            return True
    except OSError:
        if startup_stats is not None:
            startup_stats["probe_misses"] += 1
        return False


def wait_for_port(port, startup_stats):
    deadline = time.time() + PYTHON_TOOL_STARTUP_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        if is_port_open(port, startup_stats):
            return True
        time.sleep(PYTHON_TOOL_STARTUP_POLL_SECONDS)
    startup_stats["startup_wait_timeouts"] += 1
    startup_stats["startup_wait_timeout_ports"].append(port)
    return False


def wait_until_port_closed(port, timeout_seconds=5):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not is_port_open(port):
            return True
        time.sleep(PYTHON_TOOL_STARTUP_POLL_SECONDS)
    return not is_port_open(port)


def consecutive_ports(start, count):
    return [start + offset for offset in range(count)]


def parse_python_tool_ports(text):
    ports = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not ports:
        raise ValueError("--python_tool_ports must contain at least one port.")
    invalid = [port for port in ports if port <= 0 or port > 65535]
    if invalid:
        raise ValueError(f"Invalid PythonTool ports: {invalid}")
    if len(set(ports)) != len(ports):
        raise ValueError(f"Duplicate PythonTool ports: {ports}")
    return ports


def allocate_python_tool_ports(requested_start, count, gap):
    if count <= 0:
        raise ValueError("python_tool_server_count must be positive.")
    if gap <= 0:
        raise ValueError("python_tool_port_gap must be positive.")

    requested_ports = consecutive_ports(requested_start, count)
    candidate_start = requested_start
    attempts = []

    for _ in range(50):
        ports = consecutive_ports(candidate_start, count)
        if ports[-1] > 65535:
            raise RuntimeError(
                f"Unable to allocate {count} PythonTool ports: candidate range "
                f"{ports[0]}-{ports[-1]} exceeds 65535."
            )

        occupied_ports = [port for port in ports if is_port_open(port)]
        attempts.append(
            {
                "start": ports[0],
                "end": ports[-1],
                "occupied_ports": occupied_ports,
            }
        )
        if not occupied_ports:
            return {
                "requested_ports": requested_ports,
                "ports": ports,
                "port_gap": gap,
                "autoshifted": ports[0] != requested_start,
                "occupied_ports_detected": sorted(
                    {
                        occupied_port
                        for attempt in attempts
                        for occupied_port in attempt["occupied_ports"]
                    }
                ),
                "attempts": attempts,
            }

        candidate_start = max(ports[-1], max(occupied_ports)) + gap

    raise RuntimeError(
        f"Unable to allocate {count} free PythonTool ports after {len(attempts)} attempts."
    )


def make_startup_stats(ports):
    return {
        "connect_timeout_seconds": PYTHON_TOOL_STARTUP_CONNECT_TIMEOUT_SECONDS,
        "wait_timeout_seconds": PYTHON_TOOL_STARTUP_WAIT_TIMEOUT_SECONDS,
        "ports_expected": len(ports),
        "ports_started": 0,
        "ports_reused": 0,
        "start_attempts": 0,
        "probe_attempts": 0,
        "probe_misses": 0,
        "startup_wait_timeouts": 0,
        "startup_wait_timeout_ports": [],
        "startup_restart_attempts": 0,
        "startup_failed_ports": [],
        "skipped": False,
    }


def startup_health_line(startup_stats):
    return (
        "python_tool_startup_health: "
        f"started={startup_stats['ports_started']}, "
        f"reused={startup_stats['ports_reused']}, "
        f"start_attempts={startup_stats['start_attempts']}, "
        f"wait_timeouts={startup_stats['startup_wait_timeouts']}, "
        f"restarts={startup_stats['startup_restart_attempts']}, "
        f"probe_misses={startup_stats['probe_misses']} / {startup_stats['probe_attempts']}, "
        f"connect_timeout={startup_stats['connect_timeout_seconds']}s, "
        f"wait_timeout={startup_stats['wait_timeout_seconds']}s"
    )


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stop_process_tree(pid):
    try:
        parent = psutil.Process(int(pid))
    except (ValueError, psutil.NoSuchProcess):
        return

    try:
        processes = parent.children(recursive=True) + [parent]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [parent]

    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(processes, timeout=5.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=5.0)


def stop_python_tool_servers(started_servers, ports):
    closed_count = 0
    for proc in reversed(started_servers):
        if proc.poll() is None:
            stop_process_tree(proc.pid)
    for port in ports:
        if wait_until_port_closed(port):
            closed_count += 1
    if started_servers:
        print(f"closed PythonTool servers: {closed_count}", flush=True)


def prepare_python_tool_servers(args, cfg):
    explicit_ports = bool(args.python_tool_ports)
    if args.python_tool_ports:
        ports = parse_python_tool_ports(args.python_tool_ports)
        port_allocation = {
            "requested_ports": ports,
            "ports": ports,
            "port_gap": args.python_tool_port_gap,
            "autoshifted": False,
            "occupied_ports_detected": [],
            "attempts": [],
        }
    elif args.reuse_existing_python_tool_servers:
        ports = consecutive_ports(
            args.python_tool_port_start,
            args.python_tool_server_count,
        )
        port_allocation = {
            "requested_ports": ports,
            "ports": ports,
            "port_gap": args.python_tool_port_gap,
            "autoshifted": False,
            "occupied_ports_detected": [],
            "attempts": [],
        }
    else:
        port_allocation = allocate_python_tool_ports(
            requested_start=args.python_tool_port_start,
            count=args.python_tool_server_count,
            gap=args.python_tool_port_gap,
        )
        ports = port_allocation["ports"]

    ports = port_allocation["ports"]
    args.python_tool_ports = ",".join(str(port) for port in ports)

    cfg["python_tool_ports"] = args.python_tool_ports
    cfg["python_tool_port_start"] = args.python_tool_port_start
    cfg["python_tool_server_count"] = args.python_tool_server_count
    cfg["python_tool_port_gap"] = args.python_tool_port_gap
    cfg["python_tool_port_autoshifted"] = port_allocation["autoshifted"]
    cfg["occupied_python_tool_ports_detected"] = port_allocation[
        "occupied_ports_detected"
    ]

    log_dir = Path(cfg["out_root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    startup_stats = make_startup_stats(ports)
    startup_stats["requested_ports"] = port_allocation["requested_ports"]
    startup_stats["actual_ports"] = ports
    startup_stats["port_autoshifted"] = port_allocation["autoshifted"]
    startup_stats["port_gap"] = port_allocation["port_gap"]
    startup_stats["port_allocation_attempts"] = port_allocation["attempts"]
    startup_stats["occupied_ports_detected"] = port_allocation[
        "occupied_ports_detected"
    ]

    requested_ports = port_allocation["requested_ports"]
    print(
        "requested_python_tool_ports: "
        f"{requested_ports[0]}-{requested_ports[-1]} ({len(requested_ports)})",
        flush=True,
    )
    if port_allocation["autoshifted"]:
        print(
            f"occupied_ports_detected: {port_allocation['occupied_ports_detected']}",
            flush=True,
        )
        print(f"python_tool_port_gap: {port_allocation['port_gap']}", flush=True)
    print(f"python_tool_ports: {ports[0]}-{ports[-1]} ({len(ports)})", flush=True)

    started_servers = []
    if args.no_start_python_tool_server:
        startup_stats["skipped"] = True
        write_json(log_dir / "startup_health.json", startup_stats)
        print(startup_health_line(startup_stats), flush=True)
        return ports, started_servers, startup_stats

    script_path = resolve_python_tool_script_path(args)
    server_script = script_path.parent / "pack_api_windows.py"
    server_work_dir = S3_ROOT / "upstream" / "progco" / "code_interpreter"
    if not server_script.is_file():
        raise FileNotFoundError(f"Cannot find PythonTool server script: {server_script}")
    if not server_work_dir.is_dir():
        raise FileNotFoundError(f"Cannot find PythonTool server cwd: {server_work_dir}")

    for port in ports:
        if is_port_open(port, startup_stats):
            if explicit_ports or args.reuse_existing_python_tool_servers:
                startup_stats["ports_reused"] += 1
                continue
            raise RuntimeError(
                f"PythonTool port is already in use: {port}. "
                "Use automatic port allocation, a different start port, or "
                "--reuse_existing_python_tool_servers."
            )

        proc = None
        for attempt in range(1, PYTHON_TOOL_STARTUP_MAX_ATTEMPTS + 1):
            mode = "w" if attempt == 1 else "a"
            python_tool_work_dir = log_dir / f"code_interpreter_workspace_{port}"
            python_tool_work_dir.mkdir(parents=True, exist_ok=True)
            (python_tool_work_dir / "upload_file").mkdir(
                parents=True,
                exist_ok=True,
            )
            allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
                       "TMPDIR", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
                       "LANG", "LC_ALL", "CONDA_PREFIX", "VIRTUAL_ENV", "LD_LIBRARY_PATH",
                       "DYLD_LIBRARY_PATH", "SPQ_PYTHON_TOOL_LOCK_DIR",
                       "SPQ_PYTHON_TOOL_MEMORY_RESERVE_RATIO"}
            server_env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
            server_env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
            server_env["CODE_INTERPRETER_WORK_DIR"] = str(
                python_tool_work_dir.resolve()
            )
            out_log = open(
                log_dir / f"code_interpreter_{port}.out.log",
                mode,
                encoding="utf-8",
            )
            err_log = open(
                log_dir / f"code_interpreter_{port}.err.log",
                mode,
                encoding="utf-8",
            )
            proc = subprocess.Popen(
                [sys.executable, str(server_script), "--port", str(port)],
                cwd=str(server_work_dir),
                stdout=out_log,
                stderr=err_log,
                env=server_env,
            )
            out_log.close()
            err_log.close()
            started_servers.append(proc)
            startup_stats["start_attempts"] += 1

            if wait_for_port(port, startup_stats):
                startup_stats["ports_started"] += 1
                break

            startup_stats["startup_restart_attempts"] += 1
            stop_process_tree(proc.pid)
            if attempt < PYTHON_TOOL_STARTUP_MAX_ATTEMPTS:
                continue

            startup_stats["startup_failed_ports"].append(port)
            write_json(log_dir / "startup_health.json", startup_stats)
            print(startup_health_line(startup_stats), flush=True)
            raise RuntimeError(f"PythonTool server did not start on port {port}.")

    write_json(log_dir / "startup_health.json", startup_stats)
    print(f"started PythonTool servers: {startup_stats['ports_started']}", flush=True)
    print(startup_health_line(startup_stats), flush=True)
    return ports, started_servers, startup_stats


def load_python_tool_module(args):
    script_path = resolve_python_tool_script_path(args)

    module = _load_python_file(
        "S5_forward_python_tool",
        script_path,
    )

    if hasattr(
        module,
        "configure_agent_retries",
    ):
        module.configure_agent_retries(
            args.agent2_execute_retries,
            args.agent7_rewrite_retries,
            args.agent_retry_wait_seconds,
        )

    if hasattr(
        module,
        "configure_python_tool_ports",
    ):
        module.configure_python_tool_ports(
            args.python_tool_ports
        )

    if hasattr(
        module,
        "configure_python_tool_execution_timeout",
    ):
        module.configure_python_tool_execution_timeout(
            args.python_tool_execution_timeout_seconds
        )

    if hasattr(
        module,
        "configure_python_tool_http_max_attempts",
    ):
        module.configure_python_tool_http_max_attempts(
            args.python_tool_http_max_attempts
        )

    if hasattr(
        module,
        "configure_python_tool_code_loop_limits",
    ):
        module.configure_python_tool_code_loop_limits(
            args.python_tool_code_soft_warning_calls,
            args.python_tool_code_hard_stop_calls,
        )

    if hasattr(
        module,
        "configure_python_tool_duplicate_limits",
    ):
        module.configure_python_tool_duplicate_limits(
            args.python_tool_duplicate_soft_warning_calls,
            args.python_tool_duplicate_hard_block_calls,
        )

    if (
        not args.skip_python_tool_dependency_setup
        and hasattr(
            module,
            "ensure_python_tool_math_dependencies",
        )
    ):
        setup_result = (
            module.ensure_python_tool_math_dependencies()
        )

        print(
            "PythonInterpreterTool dependency setup "
            f"result: {setup_result}",
            flush=True,
        )

    if (
        not args.skip_python_tool_probe
        and hasattr(
            module,
            "request_python_tool_raw",
        )
    ):
        probe = module.request_python_tool_raw(
            "def execute():\n"
            "    return 1 + 1\n"
            "execute()"
        )

        if "2" not in str(probe):
            raise RuntimeError(
                f"PythonInterpreterTool probe failed: {probe}"
            )

    return module


def build_forward_agent(
    args,
    python_tool_module=None,
    successful_tool_flow_cache=None,
):
    """
    Construct the original Forward ProgCoAgent while injecting the
    original PythonTool executor.

    This mirrors the original S3 PythonTool runner:
      - ProgCoAgent.__new__
      - inject PythonToolLLMCodeExecutor
      - inject Feebacker and logger
      - S5 supplies A0 via first_response_override

    Parallel rollout creates independent Forward agents. The already
    configured PythonTool module may be reused so ports/retries/probes
    remain centralized.
    """
    import MATH.progco_infer_main as progco

    if python_tool_module is None:
        python_tool_module = (
            load_python_tool_module(args)
        )

    run_args = SimpleNamespace(
        model=args.forward_model,
        max_cur_turn=args.max_cur_turn,
        dataset="MATH",
        temperature=args.forward_temperature,
        branch_math_model=(
            args.branch_math_model
            or args.forward_model
        ),
        disable_branch_math=(
            args.disable_branch_math
        ),
    )

    agent = progco.ProgCoAgent.__new__(
        progco.ProgCoAgent
    )

    agent.dataset = "MATH"
    agent.model = args.forward_model
    agent.max_cur_turn = args.max_cur_turn
    agent.temperature = args.forward_temperature

    agent.llm_code_executor = (
        python_tool_module
        .PythonToolLLMCodeExecutor(
            model=args.forward_model,
            temperature=args.forward_temperature,
            enable_branch_math=(
                not args.disable_branch_math
            ),
            branch_math_model=(
                args.branch_math_model
                or args.forward_model
            ),
            successful_tool_flow_cache=(
                successful_tool_flow_cache
            ),
        )
    )

    agent.feedbacker = progco.Feebacker(
        args.forward_model,
        args.forward_temperature,
    )

    agent.logger = S5MemoryLogger(
        run_args
    )

    agent.first_response_data = {}

    return agent


def build_a0_chat_fn(args):
    """
    Return function(messages) -> raw A0 response y.

    Reuse Forward's original call_llm backend.
    """
    from utils.call_llm import call_llm

    model = (
        args.a0_model
        or args.forward_model
    )

    def chat_fn(
        messages: List[Dict[str, str]],
    ) -> str:
        response = call_llm(
            model=model,
            messages=messages,
            temperature=args.a0_temperature,
        )

        return str(
            response
            or ""
        ).strip()

    return chat_fn


def build_answer_extractor(args):
    """
    Return function(query, response) -> answer.

    Reuse Forward A8 without changing its prompt.
    """
    from extract_answer import extract_answer

    model = (
        args.answer_extractor_model
        or args.forward_model
    )

    def extractor(
        query: str,
        response: str,
    ) -> str:
        return str(
            extract_answer(
                query,
                response,
                model=model,
            )
        )

    return extractor


def _infer_dataset_key(
    dataset_path: Path,
) -> str:
    parent_name = dataset_path.parent.name

    special_names = {
        "AMO-Bench": "amo_bench",
    }

    return special_names.get(
        parent_name,
        parent_name,
    )


def _s3_api_profiles():
    module = importlib.import_module(
        "utils.call_llm"
    )

    api_profiles = getattr(
        module,
        "_api_profiles",
        None,
    )

    if api_profiles is None:
        raise RuntimeError(
            "S3 utils.call_llm did not expose _api_profiles."
        )

    return api_profiles


def _s3_model_file_tag(
    model_name: str,
) -> str:
    api_profiles = _s3_api_profiles()

    if hasattr(
        api_profiles,
        "model_file_tag",
    ):
        return str(
            api_profiles.model_file_tag(
                model_name
            )
        )

    return model_name


def _s3_api_profile_tag() -> str:
    api_profiles = _s3_api_profiles()

    if hasattr(
        api_profiles,
        "api_profile_tag",
    ):
        return str(
            api_profiles.api_profile_tag()
        )

    api_config = getattr(
        importlib.import_module(
            "utils.call_llm"
        ),
        "_api_config",
        {},
    )

    return str(
        api_config.get(
            "profile",
            "api",
        )
    )


def _default_initial_response_dir(
    model_name: str,
) -> Path:
    return (
        ROOT.parent
        / "0-1_initial_response"
        / (
            f"{_s3_api_profile_tag()}_"
            f"{_s3_model_file_tag(model_name)}"
        )
    )


def _read_raw_json_or_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            for line_index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Initial response row {line_index + 1} "
                        "must be a JSON object."
                    )
                rows.append(item)
        return rows

    if path.suffix.lower() == ".json":
        data = json.loads(
            path.read_text(
                encoding="utf-8-sig",
            )
        )
        if isinstance(data, dict):
            for key in (
                "data",
                "items",
                "records",
                "logs",
            ):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [
                        row
                        for row in rows
                        if isinstance(row, dict)
                    ]
        if isinstance(data, list):
            return [
                row
                for row in data
                if isinstance(row, dict)
            ]

    raise ValueError(
        f"Unsupported initial response file: {path}"
    )


def _load_shared_initial_response_map(
    *,
    path: Path,
    dataset_path: Path,
    dataset_key: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing shared initial response file: {path}"
        )

    rows = _read_raw_json_or_jsonl(path)
    parsed_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        sample_id = str(
            row.get("id")
            or ""
        )
        if not sample_id:
            raise ValueError(
                f"Initial response row {index + 1} is missing id."
            )

        if "response" not in row or "pred" not in row:
            raise ValueError(
                f"Initial response row {index + 1} for {sample_id!r} "
                "must contain response and pred."
            )

        parsed_rows.append(
            {
                "id": sample_id,
                "query": (
                    row.get("query")
                    or row.get("q")
                    or row.get("problem")
                ),
                "response": row["response"],
                "pred": row["pred"],
            }
        )

    initial_map: dict[str, dict[str, Any]] = {
        row["id"]: {
            "id": row["id"],
            "response": row["response"],
            "pred": row["pred"],
        }
        for row in parsed_rows
    }

    dataset_items = load_json_or_jsonl(
        dataset_path,
        dataset_key=dataset_key,
    )
    missing = [
        item["id"]
        for item in dataset_items
        if item["id"] not in initial_map
    ]
    invalid = [
        item["id"]
        for item in dataset_items
        if item["id"] in initial_map
        and (
            "response" not in initial_map[item["id"]]
            or "pred" not in initial_map[item["id"]]
        )
    ]

    if missing or invalid:
        raise ValueError(
            "Shared initial responses are incomplete: "
            f"{path}; missing={missing[:10]}, invalid={invalid[:10]}"
        )

    print(
        "  [shared initial responses] "
        f"loaded {len(initial_map)} rows from {path}",
        flush=True,
    )

    return initial_map


def resolve_shared_initial_response(
    cfg: dict,
    args,
) -> dict[str, dict[str, Any]]:
    model_name = str(
        args.a0_model
        or args.forward_model
    )
    dataset_path = Path(
        cfg["dataset_path"]
    )
    dataset_key = (
        args.dataset_key
        or _infer_dataset_key(
            dataset_path
        )
    )

    initial_dir = (
        _resolve_path(
            args.initial_response_dir,
            base_dirs=(
                Path.cwd(),
                ROOT,
                ROOT.parent,
            ),
            fallback_base=ROOT.parent,
            prefer_existing=True,
        )
        if args.initial_response_dir
        else _default_initial_response_dir(
            model_name
        )
    )

    model_tag = _s3_model_file_tag(
        model_name
    )
    initial_path = (
        initial_dir
        / f"{model_tag}_{dataset_key}_first_response.jsonl"
    )

    cfg["dataset_key"] = dataset_key
    cfg["shared_initial_response_dir"] = str(
        initial_dir
    )
    cfg["shared_initial_response_path"] = str(
        initial_path
    )

    bootstrap_shared_initial_response = _truthy(
        cfg.get(
            "bootstrap_shared_initial_response",
            True,
        )
    )
    cfg["shared_initial_response_baseline_only"] = (
        not bootstrap_shared_initial_response
    )
    cfg["shared_initial_response_for_bootstrap"] = (
        bootstrap_shared_initial_response
    )

    return _load_shared_initial_response_map(
        path=initial_path,
        dataset_path=dataset_path,
        dataset_key=dataset_key,
    )


def _redact_secret(value: str) -> str:
    return "[configured]" if value else ""


def _resolve_s3_api_runtime(
    model_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """
    Resolve the actual OpenAI-compatible API settings used by S3.

    S3 centralizes provider details in utils.call_llm and api_profiles.py.
    B0/SkillOpt must reuse the same base_url, api_key, model mapping,
    and provider-specific extra_body instead of silently falling back to
    its own optimizer default.
    """
    module = importlib.import_module(
        "utils.call_llm"
    )

    api_config = dict(
        getattr(
            module,
            "_api_config",
            {},
        )
        or {}
    )

    api_profiles = getattr(
        module,
        "_api_profiles",
        None,
    )

    if api_profiles is None:
        raise RuntimeError(
            "S3 utils.call_llm did not expose _api_profiles. "
            "Cannot synchronize B0/SkillOpt model settings."
        )

    if hasattr(
        api_profiles,
        "is_openai_compatible_model",
    ):
        if not api_profiles.is_openai_compatible_model(
            model_name
        ):
            raise RuntimeError(
                "S5 currently expects Forward/S3 to use an "
                "OpenAI-compatible API model so B0/SkillOpt can reuse "
                "the same API profile. "
                f"Got forward_model={model_name!r}."
            )

    api_model = model_name

    if hasattr(
        api_profiles,
        "api_model_name",
    ):
        api_model = str(
            api_profiles.api_model_name(
                model_name,
                api_config,
            )
        )

    extra_body = None

    if hasattr(
        api_profiles,
        "extra_body_for_model",
    ):
        try:
            extra_body = api_profiles.extra_body_for_model(
                model_name,
                api_config,
                api_model=api_model,
            )

        except TypeError:
            extra_body = api_profiles.extra_body_for_model(
                model_name,
                api_config,
            )

        if extra_body is not None and not isinstance(
            extra_body,
            dict,
        ):
            raise RuntimeError(
                "S3 api_profiles.extra_body_for_model must return "
                f"dict or None, got {type(extra_body).__name__}."
            )

    return api_model, api_config, extra_body


def sync_skillopt_model_with_s3_api(
    cfg: dict,
    args,
) -> None:
    """
    Make B0/SkillOpt use the same OpenAI-compatible API as S3.

    Forward/A0 already call S3's utils.call_llm. Reflect/Aggregate/Select
    call B0's skillopt.model.chat_optimizer. If B0 keeps its default
    optimizer_model such as gpt-5.5 while S3 is using DeepSeek, Reflect will
    call a different or non-existent model and no Skill patch will be made.
    """
    model_name = str(
        args.forward_model
        or ""
    ).strip()

    if not model_name:
        raise RuntimeError(
            "forward_model is empty; cannot synchronize SkillOpt model."
        )

    api_model, api_config, extra_body = _resolve_s3_api_runtime(
        model_name
    )

    base_url = str(
        api_config.get(
            "base_url",
            "",
        )
        or ""
    ).strip()

    api_key = str(
        api_config.get(
            "api_key",
            "",
        )
        or ""
    ).strip()

    if not base_url:
        raise RuntimeError(
            "S3 api_profiles.get_api_config() did not provide base_url. "
            "B0/SkillOpt cannot be synchronized with the Forward API."
        )

    # Use B0's existing OpenAI-compatible path. Do not introduce a new
    # DeepSeek-specific backend: DeepSeek exposes an OpenAI-compatible
    # /chat/completions API, and S3 already supplies provider details.
    cfg["model_backend"] = "azure_openai"
    cfg["optimizer_backend"] = "openai_chat"
    cfg["target_backend"] = "openai_chat"

    cfg["optimizer_model"] = api_model
    cfg["target_model"] = api_model

    cfg["azure_openai_endpoint"] = base_url
    cfg["azure_openai_api_key"] = api_key
    cfg["azure_openai_auth_mode"] = "openai_compatible"

    cfg["optimizer_azure_openai_endpoint"] = base_url
    cfg["optimizer_azure_openai_api_key"] = api_key
    cfg["optimizer_azure_openai_auth_mode"] = "openai_compatible"

    cfg["target_azure_openai_endpoint"] = base_url
    cfg["target_azure_openai_api_key"] = api_key
    cfg["target_azure_openai_auth_mode"] = "openai_compatible"

    # Let S3's provider profile decide provider-specific parameters through
    # extra_body. Avoid leaking OpenAI-only reasoning_effort into generic
    # OpenAI-compatible providers.
    cfg["reasoning_effort"] = ""
    os.environ["SKILLOPT_OPENAI_TEMPERATURE"] = str(
        float(args.forward_temperature)
    )

    # Keep fail-fast behavior on by default unless the config explicitly sets it.
    cfg.setdefault(
        "optimizer_probe",
        True,
    )

    if extra_body:
        os.environ["SKILLOPT_OPENAI_EXTRA_BODY"] = json.dumps(
            extra_body,
            ensure_ascii=False,
        )
    else:
        os.environ.pop(
            "SKILLOPT_OPENAI_EXTRA_BODY",
            None,
        )

    print(
        "  [S5 model sync] B0/SkillOpt uses S3 API profile: "
        f"forward_model={model_name} -> api_model={api_model}, "
        f"base_url={base_url}, api_key={_redact_secret(api_key)}, "
        f"extra_body={'yes' if extra_body else 'no'}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defer_report_scoring", action="store_true",
                        help="Evaluate saved answers with the separate unified evaluator.")

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
    )

    parser.add_argument(
        "--dataset_key",
        default="",
    )

    parser.add_argument(
        "--initial_response_dir",
        default="",
    )

    parser.add_argument(
        "--skill_init",
        default=None,
    )

    parser.add_argument(
        "--out_root",
        default="",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--forward_model",
        default="dso_DSv4Flash",
    )

    parser.add_argument(
        "--a0_model",
        default="",
    )

    parser.add_argument(
        "--answer_extractor_model",
        default="",
    )

    parser.add_argument(
        "--branch_math_model",
        default="",
    )

    parser.add_argument(
        "--max_cur_turn",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--forward_temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--a0_temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--python_tool_script",
        default="",
    )

    parser.add_argument(
        "--python_tool_ports",
        default="",
    )

    parser.add_argument(
        "--python_tool_port_start",
        type=int,
        default=4200,
    )

    parser.add_argument(
        "--python_tool_port_gap",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--python_tool_server_count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--python_tool_execution_timeout_seconds",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--python_tool_http_max_attempts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--python_tool_code_soft_warning_calls",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--python_tool_code_hard_stop_calls",
        type=int,
        default=70,
    )

    parser.add_argument(
        "--python_tool_duplicate_soft_warning_calls",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--python_tool_duplicate_hard_block_calls",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--agent2_execute_retries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--agent7_rewrite_retries",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--agent_retry_wait_seconds",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--llm_timeout_seconds",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--llm_max_retries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--llm_stop_max_retries",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--disable_branch_math",
        action="store_true",
    )

    parser.add_argument(
        "--skip_python_tool_probe",
        action="store_true",
    )

    parser.add_argument(
        "--skip_python_tool_dependency_setup",
        action="store_true",
    )

    parser.add_argument(
        "--no_start_python_tool_server",
        action="store_true",
    )

    parser.add_argument(
        "--reuse_existing_python_tool_servers",
        action="store_true",
    )

    return parser.parse_args()


def build_cfg(args) -> dict:
    config_path = _resolve_path(
        args.config,
        base_dirs=(
            Path.cwd(),
            ROOT,
            B0_ROOT,
        ),
        fallback_base=Path.cwd(),
        prefer_existing=True,
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot find SkillOpt config: {config_path}"
        )

    raw_cfg = load_config(
        str(config_path),
        overrides=args.cfg_options,
    )

    compression_cfg = raw_cfg.get(
        "compression",
        {},
    )

    if compression_cfg is None:
        compression_cfg = {}

    if not isinstance(
        compression_cfg,
        dict,
    ):
        raise TypeError(
            "compression config must be a mapping."
        )

    cfg = (
        flatten_config(raw_cfg)
        if is_structured(raw_cfg)
        else dict(raw_cfg)
    )

    # flatten_config() handles only the original SkillOpt sections.
    # Preserve the independent C0 section without changing the shared loader.
    cfg["compression"] = dict(
        compression_cfg
    )

    dataset_path = _resolve_path(
        args.dataset_path,
        base_dirs=(
            Path.cwd(),
            ROOT,
        ),
        fallback_base=Path.cwd(),
        prefer_existing=True,
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Cannot find forward_math dataset: {dataset_path}"
        )

    cfg["env"] = "forward_math"
    cfg["dataset_path"] = str(
        dataset_path
    )
    cfg["data_path"] = str(
        dataset_path
    )

    if args.skill_init:
        cfg["skill_init"] = (
            args.skill_init
        )

    use_initial_skill = _truthy(
        cfg.get(
            "use_initial_skill",
            False,
        )
    )
    cfg["use_initial_skill"] = use_initial_skill

    if use_initial_skill and not cfg.get("skill_init"):
        cfg["skill_init"] = (
            "skillopt/envs/forward_math/"
            "skills/initial.json"
        )

    if not cfg.get(
        "rollout_system_path"
    ):
        cfg["rollout_system_path"] = (
            "skillopt/envs/forward_math/"
            "prompts/rollout_system.md"
        )

    if use_initial_skill:
        cfg["skill_init"] = _resolve_under_B0(
            cfg["skill_init"]
        )

    cfg["rollout_system_path"] = (
        _resolve_under_B0(
            cfg["rollout_system_path"]
        )
    )

    if use_initial_skill and not Path(
        cfg["skill_init"]
    ).is_file():
        raise FileNotFoundError(
            f"Cannot find initial skill: {cfg['skill_init']}"
        )

    if not Path(
        cfg["rollout_system_path"]
    ).is_file():
        raise FileNotFoundError(
            "Cannot find forward_math rollout prompt: "
            f"{cfg['rollout_system_path']}"
        )

    cfg.setdefault(
        "bootstrap_shared_initial_response",
        True,
    )
    cfg["bootstrap_shared_initial_response"] = _truthy(
        cfg.get(
            "bootstrap_shared_initial_response",
            True,
        )
    )

    cfg.setdefault(
        "skill_update_mode",
        "patch",
    )

    cfg.setdefault(
        "use_gate",
        False,
    )

    cfg.setdefault(
        "gate_metric",
        "hard",
    )

    cfg.setdefault(
        "eval_test",
        False,
    )

    cfg.setdefault(
        "use_slow_update",
        False,
    )

    cfg.setdefault(
        "use_meta_skill",
        False,
    )

    cfg.setdefault(
        "lr_control_mode",
        "fixed",
    )

    cfg.setdefault(
        "edit_budget",
        4,
    )

    cfg.setdefault(
        "min_edit_budget",
        1,
    )

    sync_skillopt_model_with_s3_api(
        cfg,
        args,
    )

    if args.out_root:
        cfg["out_root"] = (
            args.out_root
        )

    if not cfg.get("out_root"):
        env_name = str(
            cfg.get(
                "env",
                "forward_math",
            )
            or "forward_math"
        )

        model_name = str(
            cfg.get("optimizer_model")
            or args.forward_model
            or "unknown"
        )

        for token in (
            "/",
            "\\",
            ":",
            " ",
        ):
            model_name = (
                model_name.replace(
                    token,
                    "-",
                )
            )

        timestamp = (
            datetime.datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        cfg["out_root"] = str(
            (
                ROOT
                / "outputs"
                / (
                    f"skillopt_{env_name}_"
                    f"{model_name}_{timestamp}"
                )
            ).resolve()
        )

    else:
        cfg["out_root"] = str(
            _resolve_path(
                cfg["out_root"],
                base_dirs=(
                    Path.cwd(),
                    ROOT,
                ),
                fallback_base=ROOT,
                prefer_existing=False,
            )
        )

    # ReflACTTrainer writes cfg to config.json.
    # Runtime objects must never be stored in cfg.
    try:
        json.dumps(cfg)

    except TypeError as error:
        raise TypeError(
            "S5 trainer config must contain only "
            "JSON-serializable values. Runtime objects "
            "such as ForwardSkillRunner must be passed "
            "through the adapter constructor."
        ) from error

    return cfg


def _positive_int_cfg(
    value,
    *,
    name: str,
    default: int = 1,
) -> int:
    try:
        parsed = int(
            value
            if value not in (None, "")
            else default
        )

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive integer."
        ) from error

    if parsed < 1:
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return parsed


def sync_python_tool_server_count_with_rollout_workers(
    args,
    cfg: dict,
) -> None:
    """
    S5-2 rollout_workers controls sample-level Forward rollout parallelism.

    If the user did not explicitly provide --python_tool_ports, make the
    PythonTool server count at least rollout_workers so the default parallel
    rollout has matching tool capacity. Explicit ports are respected as-is.
    """
    rollout_workers = _positive_int_cfg(
        cfg.get(
            "rollout_workers",
            1,
        ),
        name="rollout_workers",
        default=1,
    )

    cfg["rollout_workers"] = rollout_workers

    if args.python_tool_ports:
        port_count = len(
            parse_python_tool_ports(
                args.python_tool_ports
            )
        )

        if port_count < rollout_workers:
            print(
                "[S5 rollout parallel] "
                f"rollout_workers={rollout_workers}, "
                f"python_tool_ports={port_count}. "
                "LLM rollout remains parallel, but PythonTool calls may "
                "queue on the provided ports.",
                flush=True,
            )

        return

    args.python_tool_server_count = max(
        int(args.python_tool_server_count or 1),
        rollout_workers,
    )


def main():
    args = parse_args()

    os.environ[
        "SPQ_LLM_TIMEOUT_SECONDS"
    ] = str(
        args.llm_timeout_seconds
    )

    os.environ[
        "SPQ_LLM_MAX_RETRIES"
    ] = str(
        args.llm_max_retries
    )

    os.environ[
        "SPQ_LLM_STOP_MAX_RETRIES"
    ] = str(
        args.llm_stop_max_retries
    )

    os.environ.setdefault(
        "PROJECT_PATH",
        str(S3_PROGCO_ROOT),
    )

    cfg = build_cfg(args)
    if args.defer_report_scoring and cfg.get("strict_validation_metric") != "forward_pass_rate":
        raise ValueError("Deferred evaluation requires selection by forward_pass_rate.")
    if os.environ.get("SPQ_COMPRESSION_CAPACITIES_PATH"):
        cfg["compression"]["capacities_path"] = os.environ["SPQ_COMPRESSION_CAPACITIES_PATH"]
    cfg["defer_report_scoring"] = args.defer_report_scoring

    token_length_record_path = (
        Path(cfg["out_root"]) / "token_length_exceed_records.jsonl"
    )
    os.environ["SPQ_TOKEN_LENGTH_RECORD_PATH"] = str(
        token_length_record_path
    )
    os.environ["SKILLOPT_TOKEN_LENGTH_RECORD_PATH"] = str(
        token_length_record_path
    )

    api_model, api_config, extra_body = (
        _resolve_s3_api_runtime(
            args.forward_model
        )
    )

    call_llm_module = importlib.import_module(
        "utils.call_llm"
    )

    api_profiles = getattr(
        call_llm_module,
        "_api_profiles",
        None,
    )

    if api_profiles is None:
        raise RuntimeError(
            "Cannot configure context compression: "
            "utils.call_llm did not expose _api_profiles."
        )

    cfg["compression_runtime"] = (
        configure_context_runtime(
            config=cfg.get(
                "compression",
                {},
            ),
            output_root=cfg["out_root"],
            api_profiles=api_profiles,
            work_model_alias=(
                args.forward_model
            ),
            work_api_model=api_model,
            api_config=api_config,
            work_extra_body=extra_body,
        )
    )

    sync_python_tool_server_count_with_rollout_workers(
        args,
        cfg,
    )

    shared_initial_response_map = (
        resolve_shared_initial_response(
            cfg,
            args,
        )
    )

    ports, started_servers, _startup_stats = prepare_python_tool_servers(
        args,
        cfg,
    )

    try:
        python_tool_module = (
            load_python_tool_module(args)
        )

        a0_chat_fn = (
            build_a0_chat_fn(args)
        )

        answer_extractor = (
            build_answer_extractor(args)
        )

        successful_tool_flow_cache = {}

        def make_forward_runner() -> ForwardSkillRunner:
            forward_agent = (
                build_forward_agent(
                    args,
                    python_tool_module=(
                        python_tool_module
                    ),
                    successful_tool_flow_cache=(
                        successful_tool_flow_cache
                    ),
                )
            )

            return ForwardSkillRunner(
                progco_agent=forward_agent,
                a0_chat_fn=a0_chat_fn,
                answer_extractor=answer_extractor,
                skill_version="S_t",
            )

        runner = make_forward_runner()

        # Do not put runner or runner_factory in cfg.
        # ReflACTTrainer serializes cfg to config.json.
        adapter = ForwardMathAdapter(
            defer_report_scoring=args.defer_report_scoring,
            dataset_path=cfg.get(
                "dataset_path",
                "",
            ),
            data_path=cfg.get(
                "data_path",
                "",
            ),
            rollout_system_path=cfg.get(
                "rollout_system_path",
                "",
            ),
            forward_runner=runner,
            forward_runner_factory=(
                make_forward_runner
            ),
            rollout_workers=cfg.get(
                "rollout_workers",
                1,
            ),
            analyst_workers=cfg.get(
                "analyst_workers",
                16,
            ),
            failure_only=cfg.get(
                "failure_only",
                False,
            ),
            minibatch_size=cfg.get(
                "minibatch_size",
                8,
            ),
            edit_budget=cfg.get(
                "edit_budget",
                4,
            ),
            seed=cfg.get(
                "seed",
                42,
            ),
            limit=cfg.get(
                "limit",
                0,
            ),
            dataset_key=cfg.get(
                "dataset_key",
                "",
            ),
            metrics_model=args.forward_model,
            shared_initial_response_map=(
                shared_initial_response_map
            ),
            bootstrap_shared_initial_response=cfg.get(
                "bootstrap_shared_initial_response",
                True,
            ),
        )

        trainer = ReflACTTrainer(
            cfg=cfg,
            adapter=adapter,
        )

        trainer.train()

    finally:
        if not args.reuse_existing_python_tool_servers:
            stop_python_tool_servers(
                started_servers,
                ports,
            )


if __name__ == "__main__":
    main()
