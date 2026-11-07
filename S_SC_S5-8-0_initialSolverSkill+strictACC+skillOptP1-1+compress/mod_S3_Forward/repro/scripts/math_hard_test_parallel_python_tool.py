# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import requests


if os.name == "nt":
    import msvcrt
else:
    import fcntl


try:
    from mod_C0_compress import maybe_compress_messages
except ImportError:
    def maybe_compress_messages(
        *,
        stage,
        messages,
        pinned_prefix_count,
    ):
        return messages


PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS = 120
PYTHON_TOOL_HTTP_MAX_ATTEMPTS = 4
PYTHON_TOOL_CODE_SOFT_WARNING_CALLS = 50
PYTHON_TOOL_CODE_HARD_STOP_CALLS = 70
PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS = 10
PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS = 20
VERIFICATION_UNAVAILABLE_DUPLICATE_LOOP = (
    "verification_unavailable_duplicate_loop"
)
VERIFICATION_UNAVAILABLE_TOOL_FLOW_LOOP_EXHAUSTED = (
    "verification_unavailable_tool_flow_loop_exhausted"
)
PYTHON_TOOL_BUSY_SLEEP_SECONDS = 0.05
PYTHON_TOOL_MEMORY_RESERVE_RATIO = float(
    os.environ.get("SPQ_PYTHON_TOOL_MEMORY_RESERVE_RATIO", "0.25")
)
if not 0 <= PYTHON_TOOL_MEMORY_RESERVE_RATIO < 1:
    raise ValueError("SPQ_PYTHON_TOOL_MEMORY_RESERVE_RATIO must be in [0, 1).")
PYTHON_TOOL_PORTS = [4200]
PYTHON_TOOL_PORT_INDEX = 0
PYTHON_TOOL_PORT_LOCK = threading.Lock()
PYTHON_TOOL_LOCK_DIR = Path(os.environ.get("TEMP") or os.getcwd()) / "forward_python_tool_locks"
AGENT2_EXECUTE_RETRIES = 10
AGENT7_REWRITE_RETRIES = 10
AGENT_RETRY_WAIT_SECONDS = 10


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_runtime_event(event):
    payload = dict(event)
    payload.setdefault("time", now_text())
    payload.setdefault("pid", os.getpid())
    print("[Runtime Event] " + json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--dataset_key", default="aime_2024")
    p.add_argument("--model", default="dso_DSv4Flash")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--max_cur_turn", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1024)
    p.add_argument("--llm_timeout_seconds", type=int, default=120)
    p.add_argument("--llm_max_retries", type=int, default=5)
    p.add_argument("--llm_stop_max_retries", type=int, default=4)
    p.add_argument("--agent2_execute_retries", type=int, default=10)
    p.add_argument("--agent7_rewrite_retries", type=int, default=10)
    p.add_argument("--agent_retry_wait_seconds", type=float, default=10.0)
    p.add_argument("--python_tool_ports", default="4200")
    p.add_argument("--python_tool_execution_timeout_seconds", type=int, default=120)
    p.add_argument(
        "--python_tool_http_max_attempts",
        type=int,
        default=4,
    )
    p.add_argument("--python_tool_code_soft_warning_calls", type=int, default=50)
    p.add_argument("--python_tool_code_hard_stop_calls", type=int, default=70)
    p.add_argument(
        "--python_tool_duplicate_soft_warning_calls",
        type=int,
        default=10,
    )
    p.add_argument(
        "--python_tool_duplicate_hard_block_calls",
        type=int,
        default=20,
    )
    p.add_argument("--initial_response_dir", default=None)
    p.add_argument("--output_path", default=None)
    p.add_argument("--skip_python_tool_probe", action="store_true")
    p.add_argument("--disable_branch_math", action="store_true")
    p.add_argument("--branch_math_model", default="")
    return p.parse_args()


def load_api_profiles(public_root):
    path = public_root / "0-4_API" / "0_curr_API" / "api_profiles.py"
    spec = importlib.util.spec_from_file_location("spq_api_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATH_ARG_KEYS = {"repo_root", "data_path", "initial_response_dir", "output_path"}


def display_path(value, root):
    if value in (None, ""):
        return value
    try:
        path = Path(value)
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(value)


def display_args(args, root):
    data = vars(args).copy()
    for key in PATH_ARG_KEYS:
        if key in data:
            data[key] = display_path(data[key], root)
    return data


def configure_python_tool_ports(port_text):
    global PYTHON_TOOL_PORTS
    ports = [int(x.strip()) for x in str(port_text).split(",") if x.strip()]
    PYTHON_TOOL_PORTS = ports or [4200]


def configure_python_tool_execution_timeout(timeout_seconds):
    global PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS
    PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS = max(1, int(timeout_seconds))


def configure_python_tool_http_max_attempts(max_attempts):
    global PYTHON_TOOL_HTTP_MAX_ATTEMPTS
    PYTHON_TOOL_HTTP_MAX_ATTEMPTS = max(1, int(max_attempts))


def configure_python_tool_code_loop_limits(soft_warning_calls, hard_stop_calls):
    global PYTHON_TOOL_CODE_SOFT_WARNING_CALLS, PYTHON_TOOL_CODE_HARD_STOP_CALLS
    hard_stop = max(1, int(hard_stop_calls))
    soft_warning = max(1, min(int(soft_warning_calls), hard_stop))
    PYTHON_TOOL_CODE_SOFT_WARNING_CALLS = soft_warning
    PYTHON_TOOL_CODE_HARD_STOP_CALLS = hard_stop


def configure_python_tool_duplicate_limits(
    soft_warning_calls,
    hard_block_calls,
):
    global PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS
    global PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS

    soft_warning = int(soft_warning_calls)
    hard_block = int(hard_block_calls)

    if not 0 < soft_warning < hard_block:
        raise ValueError(
            "PythonTool duplicate limits must satisfy "
            "0 < soft_warning_calls < hard_block_calls."
        )

    PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS = soft_warning
    PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS = hard_block


def configure_agent_retries(agent2_execute_retries, agent7_rewrite_retries, agent_retry_wait_seconds):
    global AGENT2_EXECUTE_RETRIES, AGENT7_REWRITE_RETRIES, AGENT_RETRY_WAIT_SECONDS
    AGENT2_EXECUTE_RETRIES = max(1, int(agent2_execute_retries))
    AGENT7_REWRITE_RETRIES = max(1, int(agent7_rewrite_retries))
    AGENT_RETRY_WAIT_SECONDS = max(0, float(agent_retry_wait_seconds))


def next_python_tool_port():
    global PYTHON_TOOL_PORT_INDEX
    with PYTHON_TOOL_PORT_LOCK:
        port = PYTHON_TOOL_PORTS[PYTHON_TOOL_PORT_INDEX % len(PYTHON_TOOL_PORTS)]
        PYTHON_TOOL_PORT_INDEX += 1
    return port


def python_tool_lock_path(port):
    return PYTHON_TOOL_LOCK_DIR / f"port_{port}.lock"


def try_acquire_python_tool_port_lock(port):
    PYTHON_TOOL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = python_tool_lock_path(port)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd, lock_path


def release_python_tool_port_lock(lock_handle):
    if lock_handle is None:
        return
    fd, _lock_path = lock_handle
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def check_active_sample_budget():
    try:
        from utils.sample_budget import check_sample_budget
    except ImportError:
        return
    check_sample_budget()


def python_tool_memory_snapshot():
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64),
                ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64),
                ("avail_extended", ctypes.c_uint64),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return status.avail_phys, status.total_phys

    fields = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemAvailable:", "MemTotal:"))
    }
    available, total = fields["MemAvailable"], fields["MemTotal"]
    root = Path("/sys/fs/cgroup")
    group = root
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            candidate = root / line[3:].lstrip("/")
            if candidate.is_dir():
                group = candidate
            break
    while True:
        limit_path = group / "memory.max"
        if limit_path.exists():
            limit = limit_path.read_text().strip()
            if limit != "max":
                used = int((group / "memory.current").read_text())
                available = min(available, max(0, int(limit) - used))
                total = min(total, int(limit))
        if group == root:
            break
        group = group.parent
    return available, total


def acquire_python_tool_port():
    busy_started_at = None
    busy_retries = 0
    memory_wait_started_at = None

    while True:
        check_active_sample_budget()
        admission_handle = None
        memory_ready = True
        if PYTHON_TOOL_MEMORY_RESERVE_RATIO > 0:
            admission_handle = try_acquire_python_tool_port_lock("memory_admission_s5")
            memory_ready = admission_handle is not None
        try:
            if admission_handle is not None:
                try:
                    available, total = python_tool_memory_snapshot()
                    reserve = min(int(total * PYTHON_TOOL_MEMORY_RESERVE_RATIO), 50 * 1024 ** 3)
                    memory_ready = available >= reserve
                    reason = "low_memory"
                except (OSError, ValueError, KeyError) as error:
                    available = None
                    reserve = None
                    memory_ready = False
                    reason = str(error)
                if not memory_ready and memory_wait_started_at is None:
                    memory_wait_started_at = time.monotonic()
                    log_runtime_event({
                        "stage": "python_tool_memory_admission", "status": "waiting",
                        "available_bytes": available,
                        "reserve_bytes": reserve,
                        "reserve_ratio": PYTHON_TOOL_MEMORY_RESERVE_RATIO,
                        "reason": reason,
                    })
            if memory_ready:
                for _ in range(len(PYTHON_TOOL_PORTS)):
                    port = next_python_tool_port()
                    lock_handle = try_acquire_python_tool_port_lock(port)
                    if lock_handle is not None:
                        busy_wait_seconds = 0.0
                        if busy_started_at is not None:
                            busy_wait_seconds = round(time.monotonic() - busy_started_at, 3)
                        if memory_wait_started_at is not None:
                            log_runtime_event({
                                "stage": "python_tool_memory_admission", "status": "resumed",
                                "wait_seconds": round(time.monotonic() - memory_wait_started_at, 3),
                            })
                        return port, lock_handle, busy_retries, busy_wait_seconds
        finally:
            release_python_tool_port_lock(admission_handle)

        busy_retries += 1
        if busy_started_at is None:
            busy_started_at = time.monotonic()
        check_active_sample_budget()
        time.sleep(1.0 if memory_wait_started_at is not None else PYTHON_TOOL_BUSY_SLEEP_SECONDS)


def normalize_hard_test_data(raw, dataset_key):
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("logs") or []
    data = []
    for i, item in enumerate(raw):
        row = dict(item)
        query = (
            row.get("query")
            or row.get("problem")
            or row.get("q")
            or row.get("question")
            or row.get("prompt")
        )
        row["id"] = f"{dataset_key}_{i}"
        row["query"] = query
        data.append(row)
    return data


def require_first_response_map(first_path, first_rows, data):
    first_map = {x["id"]: x for x in first_rows if isinstance(x, dict) and "id" in x}
    missing = [dp["id"] for dp in data if dp["id"] not in first_map]
    invalid = [
        dp["id"] for dp in data
        if dp["id"] in first_map and ("response" not in first_map[dp["id"]] or "pred" not in first_map[dp["id"]])
    ]
    if missing or invalid:
        raise ValueError(
            f"Shared initial responses are incomplete: {first_path}; "
            f"missing={missing[:10]}, invalid={invalid[:10]}"
        )
    return first_map


def extract_tool_flow(text):
    matches = re.findall(r"<tool_flow>(.*?)</tool_flow>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Robust fallback when the stop-word API returns text ending at the stop word
    # but the closing tag is not included in the returned response.
    if "<tool_flow>" in text:
        return text.split("<tool_flow>", 1)[-1].strip()

    return ""


def extract_python_code(text):
    text = text.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    text = (
        text.replace("<tool_flow>", "")
            .replace("</tool_flow>", "")
            .strip()
    )
    return text.strip()


def extract_result_text(text):
    matches = re.findall(r"<result>(.*?)</result>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    if "[Verification Result]" in text:
        return text.split("[Verification Result]", 1)[1].strip()
    return text.strip()


def classify_python_tool_result(text):
    lowered = str(text).lower()
    if "timeout. no response after" in lowered:
        return "execution_timeout"
    if "request code interpreter failed" in lowered:
        return "request_failed"
    if any(x in lowered for x in ("traceback", "syntaxerror", "nameerror", "typeerror", "valueerror", "runtimeerror")):
        return "execution_error"
    return "ok"


def request_python_tool_raw_with_event(code):
    started_at = time.monotonic()
    event = {
        "status": "ok",
        "start_time": now_text(),
        "execution_timeout_seconds": PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS,
        "port": None,
        "busy_retries": 0,
        "busy_wait_seconds": 0.0,
        "http_attempts": 0,
        "http_retries": 0,
        "code_chars": len(code or ""),
    }

    transient_attempt = 0

    while True:
        port, lock_handle, busy_retries, busy_wait_seconds = acquire_python_tool_port()
        active_started_at = time.monotonic()
        event["port"] = port
        event["busy_retries"] += busy_retries
        event["busy_wait_seconds"] = round(
            float(event["busy_wait_seconds"])
            + float(busy_wait_seconds),
            3,
        )
        retry_transient = False

        try:
            event["http_attempts"] += 1
            response = requests.post(
                f"http://localhost:{port}/api/code_interpreter",
                json={
                    "code": code,
                    "execution_timeout_seconds": PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS,
                },
                timeout=(10, PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS + 10),
            )
            if response.status_code == 429:
                event["busy_retries"] += 1
                time.sleep(PYTHON_TOOL_BUSY_SLEEP_SECONDS)
                continue

            if 500 <= response.status_code <= 599:
                transient_attempt += 1
                if transient_attempt < PYTHON_TOOL_HTTP_MAX_ATTEMPTS:
                    event["http_retries"] += 1
                    retry_transient = True
                else:
                    result_text = (
                        "request code interpreter failed: "
                        f"HTTP {response.status_code}: "
                        f"{response.text}"
                    )
            else:
                response.raise_for_status()
                payload = response.json()

                if payload.get("busy"):
                    event["busy_retries"] += 1
                    time.sleep(PYTHON_TOOL_BUSY_SLEEP_SECONDS)
                    continue

                # A successful HTTP response is an application result.
                # Do not retry execution errors or execution timeouts:
                # the submitted code may already have run.
                result = payload.get("result", "")
                result_text = (
                    str(result)
                    .replace("```", "")
                    .replace("execute_result:", "")
                    .strip()
                )

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as error:
            transient_attempt += 1
            if transient_attempt < PYTHON_TOOL_HTTP_MAX_ATTEMPTS:
                event["http_retries"] += 1
                retry_transient = True
            else:
                result_text = (
                    "request code interpreter failed: "
                    f"{type(error).__name__}: {error}"
                )

        except requests.exceptions.RequestException as error:
            result_text = (
                "request code interpreter failed: "
                f"{type(error).__name__}: {error}"
            )

        except Exception as error:
            result_text = (
                "request code interpreter failed: "
                f"{type(error).__name__}: {error}"
            )
        finally:
            release_python_tool_port_lock(lock_handle)

        if retry_transient:
            time.sleep(min(event["http_retries"], 3))
            continue

        event["request_wall_seconds"] = round(time.monotonic() - started_at, 3)
        event["active_request_seconds"] = round(time.monotonic() - active_started_at, 3)
        event["end_time"] = now_text()
        event["status"] = classify_python_tool_result(result_text)
        event["result_chars"] = len(result_text)
        return result_text, event



def request_python_tool_raw(code):
    return request_python_tool_raw_with_event(code)[0]


def ensure_python_tool_math_dependencies():
    setup_code = """import importlib.util
required = ('sympy', 'z3', 'numpy', 'scipy')
missing = [name for name in required if importlib.util.find_spec(name) is None]
print('BRANCH_MATH_DEPS_OK' if not missing else 'Install requirements.txt: ' + ', '.join(missing))"""
    result = request_python_tool_raw(setup_code)
    if "BRANCH_MATH_DEPS_OK" not in str(result):
        raise RuntimeError(f"PythonInterpreterTool dependency setup failed: {result}")
    return result


def parse_verification_result(text):
    result_text = extract_result_text(text)
    lowered = result_text.lower()
    if "false" in lowered:
        return False
    if "true" in lowered:
        return True
    return False


def extract_python_tool_events(execute_content):
    events = []
    for raw_event in re.findall(r"\[PythonTool Event\]\n(\{.*?\})(?=\n|$)", str(execute_content)):
        try:
            events.append(json.loads(raw_event))
        except Exception:
            events.append({"status": "parse_failed", "raw": raw_event})
    return events


def extract_timing_events(execute_content):
    events = []
    for raw_event in re.findall(r"\[Timing Event\]\n(\{.*?\})(?=\n|$)", str(execute_content)):
        try:
            events.append(json.loads(raw_event))
        except Exception:
            events.append({"status": "parse_failed", "raw": raw_event})
    return events


def summarize_python_tool_events(events):
    if not events:
        return "not_used"
    statuses = [str(event.get("status", "unknown")) for event in events]
    for status in ("rewrite_failed", "execution_timeout", "request_failed", "execution_error"):
        if status in statuses:
            return status
    return "ok" if all(status == "ok" for status in statuses) else "mixed"


EMPTY_CONTENT_SKIP_REASON = "empty_content_after_retries"


def is_empty_content_error(error):
    text = f"{type(error).__name__}: {str(error)}".lower()
    return "empty content" in text or EMPTY_CONTENT_SKIP_REASON in text


def build_failed_sample_log(dp, error, execute_content=""):
    python_tool_events = extract_python_tool_events(execute_content)
    timing_events = extract_timing_events(execute_content)
    empty_content = is_empty_content_error(error)
    log = {
        "id": dp.get("id"),
        "query": dp.get("query"),
        "turns": [],
        "failed": True,
        "error": f"{type(error).__name__}: {str(error)}",
        "execute_content": execute_content,
        "python_tool_events": python_tool_events,
        "timing_events": timing_events,
        "python_tool_status": summarize_python_tool_events(python_tool_events),
        "reasoning_trace_path": os.environ.get("SPQ_REASONING_TRACE_PATH", ""),
    }
    if empty_content:
        log.update({
            "skipped": True,
            "skip_reason": EMPTY_CONTENT_SKIP_REASON,
            "empty_content": True,
            "final_response": "",
            "final_result": None,
        })
    return log


class MemoryLogger:
    def __init__(self, args):
        self.args = args
        self.logs = []

    def init_sample(self, sample_id, query):
        self.logs.append({"id": sample_id, "query": query, "turns": []})

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
        python_tool_events = extract_python_tool_events(execute_content)
        timing_events = extract_timing_events(execute_content)
        python_tool_status = summarize_python_tool_events(python_tool_events)

        turn_log = {
            "cur_turn": cur_turn,
            "verify_code": verify_code,
            "response": response,
            "result": result,
            "execute_result": execute_result,
            "execute_content": execute_content,
            "python_tool_events": python_tool_events,
            "timing_events": timing_events,
            "python_tool_status": python_tool_status,
            "python_tool_execution_timeout": python_tool_status == "execution_timeout",
            "feedback": feedback,
            "reflection": reflection,
            "code_feedback": code_feedback,
        }

        # 3-1_Forward analysis fields.
        # These fields record the output of A15 and the routing decision of A16.
        if revised_response is not None:
            turn_log["revised_response"] = revised_response
        if revised_result is not None:
            turn_log["revised_result"] = revised_result
        if a16_decision is not None:
            turn_log["a16_decision"] = a16_decision
        if next_verify_source is not None:
            turn_log["next_verify_source"] = next_verify_source

        self.logs[-1]["turns"].append(turn_log)

    def update_final_response(self, response, result):
        self.logs[-1]["final_response"] = response
        self.logs[-1]["final_result"] = result

    def update_insights(self, insights):
        self.logs[-1]["insights"] = insights

    def save_log(self):
        pass


# A17 / Agent17: ToolFlow Implementer
# Implement temporary mathematical tool flows inside <tool_flow>...</tool_flow>
# as reliable executable Python without changing the verification target.
branch_math_rewrite_dict = {
    "system_prompt": r"""You are a MathTool flow implementation expert.

A verification expert wrote a mathematical tool flow inside <tool_flow>...</tool_flow>. Your task is to implement that tool flow as reliable executable Python code before execution.

Requirements:
1. Preserve the mathematical object and verification intent of the tool flow.
2. Use tool-level operations when possible: set construction, count, sum, max, min, argmax, argmin, solve, simplify, factor, recurrence evaluation, DP evaluation, emptiness checking, and counterexample search.
3. Prefer exact symbolic, rational, integer, or exhaustive computation over floating-point approximation.
4. Implement only the mathematical tool flow provided in the input.
5. Preserve the verification target exactly.
6. Use only the assumptions, constants, variables, and conditions stated in the tool flow.
7. For set, count, extremum, recurrence, symbolic, or completeness tasks, implement the full mathematical object described in the tool flow.
8. Use ASCII mathematical notation:
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
9. Output only executable Python code containing a complete execute() function and a final execute() call.
10. Use local deterministic computation with the approved mathematical libraries listed below.

Tool preference:
- Use sympy for exact symbolic algebra, simplification, expansion, factorization, square roots, radicals, polynomial expressions, symbolic equality checks, integer factorization, divisors, factorials, binomial coefficients, equation solving, resultants, Groebner bases, and exact equation manipulation.
- For symbolic equality checks, prefer simplifying the difference, such as simplify(lhs - rhs) == 0, instead of directly comparing unsimplified expressions.
- Use itertools for finite exhaustive enumeration, Cartesian products, permutations, combinations, and counting over small finite domains.
- Use fractions.Fraction for exact rational arithmetic when the computation involves fractions or ratios. Construct Fraction values from integers or strings rather than floats when possible.
- Use decimal.Decimal for decimal arithmetic and rounding when base-10 rounding rules matter. Construct Decimal values from strings, not floats. Use ROUND_HALF_UP for standard school-style rounding unless the problem states a different rule.
- Use math for simple integer utilities such as gcd, lcm, factorial, comb, perm, and isqrt when symbolic algebra is not needed.
- Use z3 only for explicit constraint satisfaction over integers, booleans, or finite logical conditions when direct enumeration or sympy is unsuitable.
- Use scipy only when the task explicitly requires numerical approximation or numerical optimization and exact methods are unsuitable.
""",

    "input_case_1": r"""[Original PythonInterpreterTool Tool Flow]
Evaluate V = {d in positive integers : d divides 1200 and d is odd}.
Return V and max(V).""",

    "output_case_1": r"""def execute():
    import sympy as sp
    V = [d for d in sp.divisors(1200) if d > 0 and d % 2 == 1]
    return {"V": V, "max": max(V)}
execute()""",

    "input_case_2": r"""[Original PythonInterpreterTool Tool Flow]
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).""",

    "output_case_2": r"""def execute():
    import itertools
    D = range(10)
    V = [(A, B) for A, B in itertools.product(D, D) if (2 + 5 + A + B) % 9 == 0]
    return {"V": V, "count": len(V)}
execute()""",

    "input_case_3": r"""[Original PythonInterpreterTool Tool Flow]
Let c be a symbolic variable.
ans := 8*c + 6
v(c) := c + 2
J(c) := c + 3*v(c)
K(c) := 5*c
T(c) := J(c) + K(c)
Return simplify(T(c) - ans).""",

    "output_case_3": r"""def execute():
    import sympy as sp
    c = sp.symbols("c")
    ans = 8 * c + 6
    v = c + 2
    J = c + 3 * v
    K = 5 * c
    T = J + K
    return sp.simplify(T - ans)
execute()""",

    "input_case_4": r"""[Original PythonInterpreterTool Tool Flow]
Given ans = 90, leg1 = 9, S = ans - leg1, and h(l) = S - l.
Evaluate V(ans) = {l in positive integers :
    h(l) in positive integers
    and abs(h(l) - l) = 1
    and leg1^2 + l^2 = h(l)^2}.
Return V(ans) and whether V(ans) is nonempty.""",

    "output_case_4": r"""def execute():
    ans = 90
    leg1 = 9
    S = ans - leg1
    V = []
    for l in range(1, S):
        h = S - l
        if h > 0 and abs(h - l) == 1 and leg1**2 + l**2 == h**2:
            V.append(l)
    return {"V": V, "nonempty": len(V) > 0}
execute()""",

    "input_case_5": r"""[Original PythonInterpreterTool Tool Flow]
Let dp_i(r) be the number of valid arrangements of the first i positions whose suffix has exactly r consecutive occupied positions, where r in {0,1,2}.
Use:
dp_0(0)=1, dp_0(1)=dp_0(2)=0.
dp_{i+1}(0)=dp_i(0)+dp_i(1)+dp_i(2)
dp_{i+1}(1)=dp_i(0)
dp_{i+1}(2)=dp_i(1)
For n = 10, return N = dp_n(0)+dp_n(1)+dp_n(2).""",

    "output_case_5": r"""def execute():
    n = 10
    dp = {0: 1, 1: 0, 2: 0}
    for _ in range(n):
        dp = {
            0: dp[0] + dp[1] + dp[2],
            1: dp[0],
            2: dp[1],
        }
    return {"dp": dp, "N": dp[0] + dp[1] + dp[2]}
execute()""",

    "input_case_6": r"""[Original PythonInterpreterTool Tool Flow]
Compute CE = sqrt(4^2 + 9^2).
Return CE exactly and rounded to the nearest tenth.""",

    "output_case_6": r"""def execute():
    import sympy as sp
    CE = sp.sqrt(4**2 + 9**2)
    return {"exact": sp.simplify(CE), "rounded_nearest_tenth": round(float(CE), 1)}
execute()""",

    "input_case_7": r"""[Original PythonInterpreterTool Tool Flow]
Let ans = 1050*sqrt(2).
Return simplify(ans^2).""",

    "output_case_7": r"""def execute():
    import sympy as sp
    ans = 1050 * sp.sqrt(2)
    return sp.simplify(ans**2)
execute()""",

    "input_case_8": r"""[Original PythonInterpreterTool Tool Flow]
O = {A, B, C}.
r(A) = round_to_nearest_tenth(14.45001)
r(B) = round_to_nearest_tenth(14.54999)
r(C) = round_to_nearest_tenth(14.44999)
m = min {r(o) for o in O}
V = {o in O : r(o) = m}
Return r, m, and V.""",

    "output_case_8": r"""def execute():
    from decimal import Decimal, ROUND_HALF_UP
    values = {
        "A": Decimal("14.45001"),
        "B": Decimal("14.54999"),
        "C": Decimal("14.44999"),
    }
    r = {
        key: value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        for key, value in values.items()
    }
    m = min(r.values())
    V = [key for key, value in r.items() if value == m]
    return {"r": r, "m": m, "V": V}
execute()""",

    "input_template": r"""[Original PythonInterpreterTool Tool Flow]
{original_code}"""
}

class BranchMathCodeRewriter:
    def __init__(self, model, temperature):
        from utils import call_llm as call_llm_module

        self.model = model
        self.temperature = temperature
        self.call_llm = call_llm_module.call_llm
        self.get_llm_call_stats = call_llm_module.get_last_call_stats
        self.last_llm_stats = {}
        self.init_messages = [
            {"role": "system", "content": branch_math_rewrite_dict["system_prompt"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_1"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_1"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_2"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_2"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_3"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_3"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_4"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_4"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_5"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_5"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_6"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_6"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_7"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_7"]},
            {"role": "user", "content": branch_math_rewrite_dict["input_case_8"]},
            {"role": "assistant", "content": branch_math_rewrite_dict["output_case_8"]},
        ]
        self.input_template = branch_math_rewrite_dict["input_template"]

    def rewrite(self, original_code):
        retries = 0
        max_retries = AGENT7_REWRITE_RETRIES
        while retries < max_retries:
            try:
                input_content = self.input_template.format(original_code=original_code)
                messages = self.init_messages + [{"role": "user", "content": input_content}]
                response = self.call_llm(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                self.last_llm_stats = self.get_llm_call_stats()
                rewritten_code = extract_python_code(response)
                if rewritten_code.strip():
                    return rewritten_code.strip()
                return ""
            except KeyboardInterrupt:
                raise
            except ValueError:
                raise
            except Exception as e:
                self.last_llm_stats = self.get_llm_call_stats()
                if retries + 1 < max_retries:
                    wait_seconds = retries * AGENT_RETRY_WAIT_SECONDS
                    print(
                        f"Someting wrong:{e}. Retrying in "
                        f"{wait_seconds} seconds...",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                retries += 1
        return ""


class PythonToolLLMCodeExecutor:
    def __init__(
        self,
        model,
        temperature,
        enable_branch_math=True,
        branch_math_model="",
        successful_tool_flow_cache=None,
    ):
        import MATH.progco_infer_main as progco
        from utils import call_llm_with_stop_words as stop_llm_module

        self.model = model
        self.temperature = temperature
        self.call_llm_with_stop_words = stop_llm_module.call_llm_with_stop_words
        self.get_main_llm_call_stats = stop_llm_module.get_last_call_stats
        self.python_tool_events = []
        self.timing_events = []
        self.sample_id = ""
        self.llm_call_index = 0
        self.python_tool_call_index = 0
        self.code_loop_warning_sent_callers = set()
        self.successful_tool_flow_cache = (
            successful_tool_flow_cache
            if successful_tool_flow_cache is not None
            else {}
        )
        self.a12_successful_tool_flows_in_execute = {}
        self.a12_last_tool_flow = ""
        self.a12_consecutive_tool_flow_count = 0
        self.verification_unavailable_reason = ""
        self.execute_content = ""
        self.working_execute_content = ""

        # A12-runtime / Agent12: Forward Verification Executor with PythonTool.
        # 3-1_Forward uses <tool_flow> as the only PythonTool trigger.
        self.end_python_tool_word = "</tool_flow>"
        self.end_result_word = "</result>"
        self.enable_branch_math = enable_branch_math
        self.branch_math_model = branch_math_model or model
        self.branch_math_temperature = 0.0
        self.branch_math_rewriter = (
            BranchMathCodeRewriter(
                model=self.branch_math_model,
                temperature=self.branch_math_temperature,
            )
            if enable_branch_math else None
        )

        # A12-runtime input template for tool-flow mathematical verification.
        self.input_template = progco.execute_verify_code_dict["python_tool_input_template"]

        # A12-runtime system prompt suffix for tool-flow mathematical verification.
        system_prompt = (
            progco.execute_verify_code_dict["system_prompt"]
            + progco.execute_verify_code_dict["python_tool_system_suffix"]
        )

        # A12 few-shot examples are loaded from execute_verify_code_dict.
        self.init_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": progco.execute_verify_code_dict["input_case_1"]},
            {"role": "assistant", "content": progco.execute_verify_code_dict["output_case_1"]},
            {"role": "user", "content": progco.execute_verify_code_dict["input_case_2"]},
            {"role": "assistant", "content": progco.execute_verify_code_dict["output_case_2"]},
        ]

    @staticmethod
    def _serialize_working_messages(
        messages,
    ):
        parts = []

        for message in messages:
            role = str(
                message.get("role")
                or "unknown"
            )

            content = str(
                message.get("content")
                or ""
            )

            parts.append(
                f"[{role}]\n{content}"
            )

        return "\n\n".join(parts)

    def _set_working_execute_content(
        self,
        final_messages=None,
    ):
        working_messages = deepcopy(
            self.messages
        )

        if final_messages:
            working_messages.extend(
                deepcopy(final_messages)
            )

        working_messages = (
            maybe_compress_messages(
                stage=(
                    "A12.execute_content"
                ),
                messages=(
                    working_messages
                ),
                pinned_prefix_count=(
                    self
                    .c0_pinned_prefix_count
                ),
            )
        )

        history_messages = (
            working_messages[
                self
                .c0_pinned_prefix_count:
            ]
        )

        self.working_execute_content = (
            self
            ._serialize_working_messages(
                history_messages
            )
        )

    def add_timing_event(self, event):
        payload = dict(event)
        payload.setdefault("sample_id", self.sample_id)
        payload.setdefault("time", now_text())
        self.timing_events.append(payload)
        self.execute_content += "\n[Timing Event]\n" + json.dumps(payload, ensure_ascii=False)

    def looks_like_execution_failure(self, text):
        lowered = str(text).lower()
        markers = [
            "traceback",
            "syntaxerror",
            "nameerror",
            "typeerror",
            "valueerror",
            "importerror",
            "modulenotfounderror",
            "zerodivisionerror",
            "runtimeerror",
            "permissionerror",
            "calledprocesserror",
            "request code interpreter failed",
        ]
        return any(marker in lowered for marker in markers)

    def _record_a12_tool_flow_request(
        self,
        normalized_tool_flow,
    ):
        normalized_tool_flow = str(
            normalized_tool_flow
            or ""
        ).strip()

        if not normalized_tool_flow:
            self.a12_last_tool_flow = ""
            self.a12_consecutive_tool_flow_count = 0
            return 0

        if normalized_tool_flow == self.a12_last_tool_flow:
            self.a12_consecutive_tool_flow_count += 1
        else:
            self.a12_last_tool_flow = normalized_tool_flow
            self.a12_consecutive_tool_flow_count = 1

        return self.a12_consecutive_tool_flow_count

    def _successful_tool_flow_cache_key(
        self,
        tool_flow,
        caller,
    ):
        sample_id = (
            ""
            if self.sample_id is None
            else str(self.sample_id)
        )

        raw_caller = (
            ""
            if caller is None
            else str(caller)
        )

        raw_tool_flow = (
            ""
            if tool_flow is None
            else str(tool_flow)
        )

        if (
            not sample_id.strip()
            or not raw_tool_flow.strip()
        ):
            return None

        branch_math_rewriter = getattr(
            self,
            "branch_math_rewriter",
            None,
        )

        branch_math_model = str(
            getattr(
                branch_math_rewriter,
                "model",
                getattr(
                    self,
                    "branch_math_model",
                    "",
                ),
            )
            or ""
        )

        branch_math_temperature = float(
            getattr(
                branch_math_rewriter,
                "temperature",
                getattr(
                    self,
                    "branch_math_temperature",
                    0.0,
                ),
            )
        )

        return (
            sample_id,
            raw_caller,
            raw_tool_flow,
            str(self.model),
            float(self.temperature),
            bool(self.enable_branch_math),
            branch_math_model,
            branch_math_temperature,
        )

    def request_raw_code_interpreter(self, code):
        return request_python_tool_raw_with_event(code)

    def request_code_interpreter(
        self,
        tool_flow,
        caller="A12",
    ):
        cache_key = (
            self
            ._successful_tool_flow_cache_key(
                tool_flow,
                caller,
            )
        )

        cached = (
            self.successful_tool_flow_cache.get(
                cache_key
            )
            if cache_key is not None
            else None
        )

        if cached is not None:
            from utils.sample_budget import (
                consume_python_tool_call,
            )

            consume_python_tool_call()

            self.python_tool_call_index += 1
            tool_call_id = (
                self.python_tool_call_index
            )

            tool_result = str(
                cached["result"]
            )

            event = {
                "status": "ok",
                "caller": caller,
                "tool_call_id": tool_call_id,
                "rewritten": bool(
                    cached.get(
                        "rewritten",
                        True,
                    )
                ),
                "fallback_for_rewrite": False,
                "cache_hit": True,
                "result_chars": len(
                    tool_result
                ),
            }

            self.python_tool_events.append(
                event
            )

            cache_event = {
                "sample_id": self.sample_id,
                "stage": "python_tool_cache",
                "caller": caller,
                "status": "hit",
                "tool_call_id": tool_call_id,
                "duration_seconds": 0.0,
                "result_chars": len(
                    tool_result
                ),
            }

            log_runtime_event(
                cache_event
            )

            self.add_timing_event(
                cache_event
            )

            return tool_result

        event_start = len(
            self.python_tool_events
        )

        tool_result = (
            self
            ._request_code_interpreter_uncached(
                tool_flow,
                caller=caller,
            )
        )

        new_events = (
            self.python_tool_events[
                event_start:
            ]
        )

        execution_event = (
            new_events[-1]
            if new_events
            else None
        )

        if (
            cache_key is not None
            and isinstance(
                execution_event,
                dict,
            )
            and execution_event.get(
                "status"
            )
            == "ok"
            and str(
                tool_result
                or ""
            ).strip()
        ):
            self.successful_tool_flow_cache[
                cache_key
            ] = {
                "result": str(
                    tool_result
                ),
                "rewritten": bool(
                    execution_event.get(
                        "rewritten",
                        False,
                    )
                ),
            }

        return tool_result

    def _request_code_interpreter_uncached(self, tool_flow, caller="A12"):
        from utils.sample_budget import consume_python_tool_call

        consume_python_tool_call()
        original_code = tool_flow
        rewritten_code = ""
        rewrite_event = None
        rewrite_failed_without_code = False

        if self.enable_branch_math and self.branch_math_rewriter is not None:
            rewrite_started_at = time.monotonic()
            log_runtime_event({
                "sample_id": self.sample_id,
                "stage": "branch_math_rewrite_llm",
                "caller": caller,
                "status": "start",
                "tool_flow_chars": len(original_code or ""),
            })
            try:
                rewritten_code = self.branch_math_rewriter.rewrite(original_code)
                if rewritten_code.strip():
                    rewrite_status = "ok"
                else:
                    rewrite_status = "empty"
                    rewrite_failed_without_code = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                rewrite_status = "error"
                rewrite_failed_without_code = True
                print(f"Someting wrong:{e}. Tool-flow rewrite failed.", flush=True)
                rewritten_code = ""

            rewrite_event = {
                "stage": "branch_math_rewrite_llm",
                "caller": caller,
                "status": rewrite_status,
                "duration_seconds": round(time.monotonic() - rewrite_started_at, 3),
                "input_tool_flow_chars": len(original_code or ""),
                "output_code_chars": len(rewritten_code or ""),
                "rewritten": bool(rewritten_code.strip()) and rewritten_code.strip() != original_code.strip(),
            }
            rewrite_event.update({
                f"llm_{key}": value
                for key, value in self.branch_math_rewriter.last_llm_stats.items()
            })
            log_runtime_event({
                "sample_id": self.sample_id,
                **rewrite_event,
            })
            self.add_timing_event(rewrite_event)
        else:
            rewrite_status = "disabled"
            rewrite_failed_without_code = True
            rewrite_event = {
                "stage": "branch_math_rewrite_llm",
                "caller": caller,
                "status": rewrite_status,
                "duration_seconds": 0.0,
                "input_tool_flow_chars": len(original_code or ""),
                "output_code_chars": 0,
                "rewritten": False,
            }
            log_runtime_event({
                "sample_id": self.sample_id,
                **rewrite_event,
            })
            self.add_timing_event(rewrite_event)

        if rewrite_failed_without_code:
            self.python_tool_call_index += 1
            tool_call_id = self.python_tool_call_index
            tool_result = (
                "branch_math_rewrite_failed: no executable Python generated from <tool_flow>."
                if rewrite_status != "disabled"
                else "branch_math_rewrite_disabled: <tool_flow> requires P7 translation before Python execution."
            )
            event = {
                "status": "rewrite_failed",
                "caller": caller,
                "tool_call_id": tool_call_id,
                "rewritten": False,
                "fallback_for_rewrite": False,
                "rewrite_event": rewrite_event,
                "result_chars": len(tool_result),
            }
            self.python_tool_events.append(event)
            log_runtime_event({
                "sample_id": self.sample_id,
                "stage": "python_tool_http",
                "caller": caller,
                "status": "rewrite_failed",
                "tool_call_id": tool_call_id,
                "code_chars": 0,
                "execution_timeout_seconds": PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS,
            })
            self.add_timing_event({
                "sample_id": self.sample_id,
                "stage": "python_tool_http",
                "caller": caller,
                "status": "rewrite_failed",
                "tool_call_id": tool_call_id,
                "duration_seconds": 0.0,
            })
            return tool_result

        self.python_tool_call_index += 1
        tool_call_id = self.python_tool_call_index
        log_runtime_event({
            "sample_id": self.sample_id,
            "stage": "python_tool_http",
            "caller": caller,
            "status": "start",
            "tool_call_id": tool_call_id,
            "code_chars": len(rewritten_code or ""),
            "execution_timeout_seconds": PYTHON_TOOL_EXECUTION_TIMEOUT_SECONDS,
        })
        tool_started_at = time.monotonic()
        tool_result, event = self.request_raw_code_interpreter(rewritten_code)
        event["tool_call_id"] = tool_call_id
        event["caller"] = caller
        event["rewritten"] = True
        event["fallback_for_rewrite"] = False
        if rewrite_event is not None:
            event["rewrite_event"] = rewrite_event
        self.python_tool_events.append(event)
        tool_timing_event = {
            "sample_id": self.sample_id,
            "stage": "python_tool_http",
            "caller": caller,
            "status": event.get("status", "unknown"),
            "tool_call_id": tool_call_id,
            "port": event.get("port"),
            "duration_seconds": round(time.monotonic() - tool_started_at, 3),
            "busy_retries": event.get("busy_retries", 0),
            "busy_wait_seconds": event.get("busy_wait_seconds", 0.0),
            "active_request_seconds": event.get("active_request_seconds"),
        }
        log_runtime_event(tool_timing_event)
        self.add_timing_event(tool_timing_event)

        return tool_result

    def run_tool_flow_for_agent(self, tool_flow, caller="A12"):
        event_start = len(self.python_tool_events)
        tool_result = self.request_code_interpreter(tool_flow, caller=caller)
        for event in self.python_tool_events[event_start:]:
            self.execute_content += "\n[PythonTool Event]\n" + json.dumps(event, ensure_ascii=False)
        self.execute_content += "\nThe execution result of PythonInterpreterTool is " + str(tool_result)
        return tool_result

    def run_refiner_with_tool_flow_loop(self, model, messages, temperature, caller, final_instruction):
        from utils import call_llm as call_llm_module

        pinned_prefix_count = len(messages)
        while True:
            messages = maybe_compress_messages(
                stage=caller,
                messages=messages,
                pinned_prefix_count=pinned_prefix_count,
            )
            self.llm_call_index += 1
            llm_call_id = self.llm_call_index
            log_runtime_event({
                "sample_id": self.sample_id,
                "stage": "refiner_llm",
                "caller": caller,
                "status": "start",
                "llm_call_id": llm_call_id,
                "messages": len(messages),
            })
            llm_started_at = time.monotonic()
            try:
                response, is_stop, gen_stop_word = self.call_llm_with_stop_words(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stop_strs=[self.end_python_tool_word],
                )
            except Exception as e:
                duration = round(time.monotonic() - llm_started_at, 3)
                log_runtime_event({
                    "sample_id": self.sample_id,
                    "stage": "refiner_llm",
                    "caller": caller,
                    "status": "error",
                    "llm_call_id": llm_call_id,
                    "duration_seconds": duration,
                    "error": f"{type(e).__name__}: {str(e)}",
                })
                raise

            duration = round(time.monotonic() - llm_started_at, 3)
            llm_stats = self.get_main_llm_call_stats()
            llm_status = "ok"
            if llm_stats.get("timeout_count", 0) > 0:
                llm_status = "ok_after_llm_timeout" if response else "llm_timeout"
            llm_event = {
                "sample_id": self.sample_id,
                "stage": "refiner_llm",
                "caller": caller,
                "status": llm_status,
                "llm_call_id": llm_call_id,
                "duration_seconds": duration,
                "is_stop": bool(is_stop),
                "gen_stop_word": gen_stop_word,
                "response_chars": len(response or ""),
            }
            llm_event.update({f"llm_{key}": value for key, value in llm_stats.items()})
            log_runtime_event(llm_event)
            self.execute_content += "\n" + response
            self.add_timing_event(llm_event)

            if is_stop and gen_stop_word == self.end_python_tool_word:
                tool_flow = extract_tool_flow(response)
                if tool_flow:
                    tool_result = self.run_tool_flow_for_agent(tool_flow, caller=caller)
                    messages = messages + [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": f"The execution result of PythonInterpreterTool is {tool_result}"},
                    ]

                    if (
                        self.python_tool_call_index >= PYTHON_TOOL_CODE_SOFT_WARNING_CALLS
                        and caller not in self.code_loop_warning_sent_callers
                    ):
                        self.code_loop_warning_sent_callers.add(caller)
                        remaining_calls = max(
                            0,
                            PYTHON_TOOL_CODE_HARD_STOP_CALLS - self.python_tool_call_index,
                        )
                        warning = (
                            f"You have already used PythonInterpreterTool {self.python_tool_call_index} times in this "
                            f"sample. Please minimize further <tool_flow> calls and use at most {remaining_calls} more "
                            "PythonInterpreterTool calls before the hard limit. Use the accumulated tool results and "
                            "produce the final required output as soon as possible.\n\n"
                            + final_instruction
                        )
                        messages = messages + [{"role": "user", "content": warning}]
                        warning_event = {
                            "stage": "python_tool_flow_loop",
                            "caller": caller,
                            "status": "soft_warning",
                            "tool_flow_calls": self.python_tool_call_index,
                            "soft_warning_calls": PYTHON_TOOL_CODE_SOFT_WARNING_CALLS,
                            "hard_stop_calls": PYTHON_TOOL_CODE_HARD_STOP_CALLS,
                        }
                        log_runtime_event({"sample_id": self.sample_id, **warning_event})
                        self.add_timing_event(warning_event)

                    messages = maybe_compress_messages(
                        stage=caller,
                        messages=messages,
                        pinned_prefix_count=pinned_prefix_count,
                    )

                    if self.python_tool_call_index >= PYTHON_TOOL_CODE_HARD_STOP_CALLS:
                        stop_event = {
                            "stage": "python_tool_flow_loop",
                            "caller": caller,
                            "status": "tool_flow_loop_exhausted",
                            "tool_flow_calls": self.python_tool_call_index,
                            "soft_warning_calls": PYTHON_TOOL_CODE_SOFT_WARNING_CALLS,
                            "hard_stop_calls": PYTHON_TOOL_CODE_HARD_STOP_CALLS,
                        }
                        log_runtime_event({"sample_id": self.sample_id, **stop_event})
                        self.add_timing_event(stop_event)
                        messages = messages + [
                            {
                                "role": "user",
                                "content": (
                                    "The hard limit for PythonInterpreterTool calls has been reached. "
                                    "Use the accumulated tool results and produce the final required output now.\n\n"
                                    + final_instruction
                                ),
                            }
                        ]
                        messages = maybe_compress_messages(
                            stage=caller,
                            messages=messages,
                            pinned_prefix_count=pinned_prefix_count,
                        )
                        return call_llm_module.call_llm(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                        )

                    continue

            return response

    def generate_with_interruption(self):
        self.messages = maybe_compress_messages(
            stage="A12",
            messages=self.messages,
            pinned_prefix_count=self.c0_pinned_prefix_count,
        )
        self.llm_call_index += 1
        llm_call_id = self.llm_call_index
        log_runtime_event({
            "sample_id": self.sample_id,
            "stage": "main_executor_llm",
            "status": "start",
            "llm_call_id": llm_call_id,
            "messages": len(self.messages),
        })
        llm_started_at = time.monotonic()
        try:
            response, is_stop, gen_stop_word = self.call_llm_with_stop_words(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                stop_strs=[
                    self.end_python_tool_word,
                    self.end_result_word,
                ],
            )
        except Exception as e:
            duration = round(time.monotonic() - llm_started_at, 3)
            log_runtime_event({
                "sample_id": self.sample_id,
                "stage": "main_executor_llm",
                "status": "error",
                "llm_call_id": llm_call_id,
                "duration_seconds": duration,
                "error": f"{type(e).__name__}: {str(e)}",
            })
            raise

        duration = round(time.monotonic() - llm_started_at, 3)
        llm_stats = self.get_main_llm_call_stats()
        llm_status = "ok"
        if llm_stats.get("timeout_count", 0) > 0:
            llm_status = "ok_after_llm_timeout" if response else "llm_timeout"
        llm_event = {
            "sample_id": self.sample_id,
            "stage": "main_executor_llm",
            "status": llm_status,
            "llm_call_id": llm_call_id,
            "duration_seconds": duration,
            "is_stop": bool(is_stop),
            "gen_stop_word": gen_stop_word,
            "response_chars": len(response or ""),
        }
        llm_event.update({f"llm_{key}": value for key, value in llm_stats.items()})
        log_runtime_event(llm_event)
        self.execute_content += "\n" + response
        self.add_timing_event(llm_event)

        if is_stop and gen_stop_word == self.end_python_tool_word:
            tool_flow = extract_tool_flow(response)
            if tool_flow:
                caller = "A12"

                normalized_tool_flow = str(
                    tool_flow
                    or ""
                ).strip()

                consecutive_request_count = (
                    self._record_a12_tool_flow_request(
                        normalized_tool_flow
                    )
                )

                successful_tool_result = (
                    self
                    .a12_successful_tool_flows_in_execute
                    .get(
                        normalized_tool_flow
                    )
                )

                if (
                    successful_tool_result is not None
                    and consecutive_request_count
                    >= PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS
                ):
                    from utils.sample_budget import (
                        consume_python_tool_call,
                    )

                    consume_python_tool_call()
                    self.python_tool_call_index += 1

                    hard_stop_message = (
                        "This identical <tool_flow> has now been requested "
                        f"{consecutive_request_count} consecutive times and "
                        "has reached the hard consecutive repeat limit of "
                        f"{PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS}. The current "
                        "verification will stop now.\n\n"
                        "The cached execution result remains:\n"
                        f"{successful_tool_result}\n\n"
                        "No further <tool_flow> requests will be accepted in "
                        "this verification."
                    )

                    hard_stop_event = {
                        "stage": "python_tool_flow_loop",
                        "caller": caller,
                        "status": (
                            "consecutive_identical_successful_tool_flow_"
                            "hard_stopped"
                        ),
                        "consecutive_request_count": (
                            consecutive_request_count
                        ),
                        "duplicate_soft_warning_calls": (
                            PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS
                        ),
                        "duplicate_hard_block_calls": (
                            PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS
                        ),
                        "tool_flow_chars": len(
                            normalized_tool_flow
                        ),
                        "tool_flow_calls": (
                            self.python_tool_call_index
                        ),
                        "cache_available": True,
                    }

                    log_runtime_event(
                        {
                            "sample_id": self.sample_id,
                            **hard_stop_event,
                        }
                    )
                    self.add_timing_event(
                        hard_stop_event
                    )

                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": hard_stop_message,
                        }
                    )

                    self.verification_unavailable_reason = (
                        VERIFICATION_UNAVAILABLE_DUPLICATE_LOOP
                    )

                    stop_content = (
                        "[Verification Unavailable]\n"
                        f"{self.verification_unavailable_reason}\n"
                        "[Consecutive Identical Tool Flow Requests]\n"
                        f"{consecutive_request_count}"
                    )

                    self.execute_content += (
                        "\n" + stop_content
                    )

                    self._set_working_execute_content(
                        final_messages=[
                            {
                                "role": "user",
                                "content": stop_content,
                            }
                        ],
                    )

                    return False, self.execute_content

                event_start = len(
                    self.python_tool_events
                )

                tool_result = (
                    self
                    .run_tool_flow_for_agent(
                        tool_flow,
                        caller=caller,
                    )
                )

                new_events = (
                    self.python_tool_events[
                        event_start:
                    ]
                )

                execution_event = (
                    new_events[-1]
                    if new_events
                    else None
                )

                if (
                    normalized_tool_flow
                    and isinstance(
                        execution_event,
                        dict,
                    )
                    and execution_event.get(
                        "status"
                    )
                    == "ok"
                    and str(
                        tool_result
                        or ""
                    ).strip()
                    and not self.looks_like_execution_failure(
                        tool_result
                    )
                ):
                    self.a12_successful_tool_flows_in_execute[
                        normalized_tool_flow
                    ] = str(
                        tool_result
                    )

                successful_tool_result = (
                    self
                    .a12_successful_tool_flows_in_execute
                    .get(
                        normalized_tool_flow
                    )
                )

                duplicate_reminder = None

                if (
                    successful_tool_result is not None
                    and consecutive_request_count
                    >= PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS
                ):
                    remaining_consecutive_requests = max(
                        0,
                        (
                            PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS
                            - consecutive_request_count
                        ),
                    )
                    duplicate_reminder = (
                        "This identical <tool_flow> has now been requested "
                        f"{consecutive_request_count} consecutive times.\n\n"
                        "The cached execution result remains:\n"
                        f"{successful_tool_result}\n\n"
                        "The hard consecutive repeat limit will be reached "
                        "after "
                        f"{remaining_consecutive_requests} more identical "
                        "requests. Do not request the same <tool_flow> again. "
                        "Use the cached result and produce "
                        "<result>True</result> or <result>False</result> as "
                        "soon as possible."
                    )
                elif (
                    successful_tool_result is not None
                    and consecutive_request_count == 2
                ):
                    duplicate_reminder = (
                        "This identical <tool_flow> has already completed "
                        "successfully.\n\n"
                        "The cached execution result is:\n"
                        f"{successful_tool_result}\n\n"
                        "Please use this result to continue the current "
                        "verification. There is no need to execute the same "
                        "<tool_flow> again."
                    )
                elif (
                    successful_tool_result is not None
                    and consecutive_request_count > 2
                ):
                    duplicate_reminder = (
                        "A successful execution result is already available "
                        "for the <tool_flow> above.\n\n"
                        "The cached execution result remains:\n"
                        f"{successful_tool_result}\n\n"
                        "Please continue the current verification using that "
                        "result. The same <tool_flow> does not need to be run "
                        "again."
                    )

                if duplicate_reminder is not None:
                    if (
                        consecutive_request_count
                        >= PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS
                    ):
                        repeat_status = (
                            "consecutive_identical_successful_tool_flow_"
                            "soft_warning"
                        )
                    elif consecutive_request_count == 2:
                        repeat_status = (
                            "consecutive_identical_successful_tool_flow_"
                            "cache_reused"
                        )
                    else:
                        repeat_status = (
                            "consecutive_identical_successful_tool_flow_"
                            "cache_reminded_again"
                        )

                    repeat_event = {
                        "stage": "python_tool_flow_loop",
                        "caller": caller,
                        "status": repeat_status,
                        "consecutive_request_count": (
                            consecutive_request_count
                        ),
                        "duplicate_soft_warning_calls": (
                            PYTHON_TOOL_DUPLICATE_SOFT_WARNING_CALLS
                        ),
                        "duplicate_hard_block_calls": (
                            PYTHON_TOOL_DUPLICATE_HARD_BLOCK_CALLS
                        ),
                        "tool_flow_chars": len(
                            normalized_tool_flow
                        ),
                        "tool_flow_calls": (
                            self.python_tool_call_index
                        ),
                        "cache_available": True,
                    }

                    log_runtime_event(
                        {
                            "sample_id": self.sample_id,
                            **repeat_event,
                        }
                    )
                    self.add_timing_event(
                        repeat_event
                    )

                self.messages.append({"role": "assistant", "content": response})
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            duplicate_reminder
                            if duplicate_reminder is not None
                            else f"The execution result of PythonInterpreterTool is {tool_result}"
                        ),
                    }
                )

                if (
                    self.python_tool_call_index >= PYTHON_TOOL_CODE_SOFT_WARNING_CALLS
                    and caller not in self.code_loop_warning_sent_callers
                ):
                    self.code_loop_warning_sent_callers.add(caller)
                    remaining_calls = max(
                        0,
                        PYTHON_TOOL_CODE_HARD_STOP_CALLS - self.python_tool_call_index,
                    )
                    warning = (
                        f"You have already used PythonInterpreterTool {self.python_tool_call_index} times in this "
                        f"verification. Please minimize further <tool_flow> calls and use at most {remaining_calls} more "
                        "PythonInterpreterTool calls before the hard limit. Use the accumulated tool results and "
                        "produce <result>True</result> or <result>False</result> as soon as possible."
                    )
                    self.messages.append({"role": "user", "content": warning})
                    warning_event = {
                        "stage": "python_tool_flow_loop",
                        "caller": caller,
                        "status": "soft_warning",
                        "tool_flow_calls": self.python_tool_call_index,
                        "soft_warning_calls": PYTHON_TOOL_CODE_SOFT_WARNING_CALLS,
                        "hard_stop_calls": PYTHON_TOOL_CODE_HARD_STOP_CALLS,
                    }
                    log_runtime_event({"sample_id": self.sample_id, **warning_event})
                    self.add_timing_event(warning_event)

                self.messages = maybe_compress_messages(
                    stage="A12",
                    messages=self.messages,
                    pinned_prefix_count=self.c0_pinned_prefix_count,
                )

                if self.python_tool_call_index >= PYTHON_TOOL_CODE_HARD_STOP_CALLS:
                    stop_event = {
                        "stage": "python_tool_flow_loop",
                        "caller": caller,
                        "status": "tool_flow_loop_exhausted",
                        "tool_flow_calls": self.python_tool_call_index,
                        "soft_warning_calls": PYTHON_TOOL_CODE_SOFT_WARNING_CALLS,
                        "hard_stop_calls": PYTHON_TOOL_CODE_HARD_STOP_CALLS,
                    }
                    log_runtime_event({"sample_id": self.sample_id, **stop_event})
                    self.add_timing_event(stop_event)

                    self.verification_unavailable_reason = (
                        VERIFICATION_UNAVAILABLE_TOOL_FLOW_LOOP_EXHAUSTED
                    )

                    stop_content = (
                        "[Verification Unavailable]\n"
                        f"{self.verification_unavailable_reason}\n"
                        "[Total Python Tool Flow Requests]\n"
                        f"{self.python_tool_call_index}"
                    )

                    self.execute_content += (
                        "\n" + stop_content
                    )

                    self._set_working_execute_content(
                        final_messages=[
                            {
                                "role": "user",
                                "content": (
                                    stop_content
                                ),
                            }
                        ],
                    )

                    return (
                        False,
                        self.execute_content,
                    )

                return self.generate_with_interruption()

        self._set_working_execute_content(
            final_messages=[
                {
                    "role": "assistant",
                    "content": response,
                }
            ],
        )

        return (
            parse_verification_result(
                response
            ),
            self.execute_content,
        )

    def execute(self, query, response, result, validate_response_fuc):
        retries = 0
        last_error = ""
        while retries < AGENT2_EXECUTE_RETRIES:
            try:
                input_content = self.input_template.format(
                    query=query,
                    response=response,
                    result=result,
                    validate_response_fuc=validate_response_fuc,
                )
                self.messages = deepcopy(self.init_messages) + [{"role": "user", "content": input_content}]
                self.c0_pinned_prefix_count = len(self.messages)
                self.execute_content = ""
                self.working_execute_content = ""
                self.python_tool_events = []
                self.timing_events = []
                self.llm_call_index = 0
                self.python_tool_call_index = 0
                self.code_loop_warning_sent_callers = set()
                self.a12_successful_tool_flows_in_execute = {}
                self.a12_last_tool_flow = ""
                self.a12_consecutive_tool_flow_count = 0
                self.verification_unavailable_reason = ""
                return self.generate_with_interruption()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if not hasattr(self, "execute_content"):
                    self.execute_content = ""
                self.execute_content += "\n[LLM Error]\n" + last_error
                if is_empty_content_error(e):
                    self.execute_content += "\n[Skip Reason]\n" + EMPTY_CONTENT_SKIP_REASON
                    raise
                if retries + 1 < AGENT2_EXECUTE_RETRIES:
                    wait_seconds = retries * AGENT_RETRY_WAIT_SECONDS
                    print(
                        f"Someting wrong:{e}. Retrying in "
                        f"{wait_seconds} seconds...",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                retries += 1
        raise RuntimeError(last_error or "Execute failed")


def main():
    args = parse_args()
    os.environ["SPQ_LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout_seconds)
    os.environ["SPQ_LLM_MAX_RETRIES"] = str(args.llm_max_retries)
    os.environ["SPQ_LLM_STOP_MAX_RETRIES"] = str(args.llm_stop_max_retries)
    configure_agent_retries(
        args.agent2_execute_retries,
        args.agent7_rewrite_retries,
        args.agent_retry_wait_seconds,
    )
    configure_python_tool_ports(args.python_tool_ports)
    configure_python_tool_execution_timeout(args.python_tool_execution_timeout_seconds)
    configure_python_tool_http_max_attempts(
        args.python_tool_http_max_attempts
    )
    configure_python_tool_code_loop_limits(
        args.python_tool_code_soft_warning_calls,
        args.python_tool_code_hard_stop_calls,
    )
    configure_python_tool_duplicate_limits(
        args.python_tool_duplicate_soft_warning_calls,
        args.python_tool_duplicate_hard_block_calls,
    )
    repo_root = Path(args.repo_root).resolve()
    project_root = repo_root.parents[1]
    public_root = project_root.parent
    public_initial_root = public_root / "0-1_initial_response"
    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else None
    os.environ.setdefault("PROJECT_PATH", str(repo_root))
    sys.path[:0] = [str(repo_root), str(repo_root / "MATH")]
    os.chdir(repo_root)

    if not args.disable_branch_math:
        setup_result = ensure_python_tool_math_dependencies()
        print(f"PythonInterpreterTool dependency setup result: {setup_result}", flush=True)

    from utils.process_file import read_file
    from utils.sample_budget import (
        SampleBudgetExceeded,
        sample_budget_scope,
    )
    import MATH.progco_infer_main as progco

    if not args.skip_python_tool_probe:
        probe = request_python_tool_raw("def execute():\n    return 1 + 1\nexecute()")
        if "2" not in str(probe):
            raise RuntimeError(f"PythonInterpreterTool probe failed: {probe}")

        if not args.disable_branch_math:
            math_probe = request_python_tool_raw(
                "def execute():\n"
                "    import sympy as sp\n"
                "    import numpy as np\n"
                "    import scipy\n"
                "    import z3\n"
                "    return {\n"
                "        'sympy_divisors': sp.divisors(12),\n"
                "        'numpy_sum': int(np.array([1, 2, 3]).sum()),\n"
                "        'scipy_version': scipy.__version__,\n"
                "        'z3_var': str(z3.Int('x')),\n"
                "    }\n"
                "execute()"
            )
            if "12" not in str(math_probe) or "6" not in str(math_probe) or "x" not in str(math_probe):
                raise RuntimeError(f"PythonInterpreterTool math probe failed: {math_probe}")

    data = normalize_hard_test_data(read_file(str(data_path)), args.dataset_key)
    end = len(data) if args.end == -1 else args.end
    data = data[args.start:end]
    workers = max(1, min(args.workers, len(data)))
    output_root = project_root / "0_generateData" / "model_outputs" / args.model
    api_profiles = load_api_profiles(public_root)
    model_tag = api_profiles.model_file_tag(args.model)
    first_dir = Path(args.initial_response_dir).resolve() if args.initial_response_dir else public_initial_root / f"{api_profiles.api_profile_tag()}_{model_tag}"

    # Shared initial responses are read-only. Tests must never modify this directory.
    # Generate them only with 0-1_initial_response/generate_initial_responses.py.
    first_path = first_dir / f"{model_tag}_{args.dataset_key}_first_response.jsonl"
    if not first_path.exists():
        raise FileNotFoundError(f"Missing shared initial response file: {first_path}")
    first_map = require_first_response_map(first_path, read_file(str(first_path)), data)

    run_args = SimpleNamespace(
        model=args.model,
        max_cur_turn=args.max_cur_turn,
        dataset="MATH",
        temperature=args.temperature,
        branch_math_model=args.branch_math_model or args.model,
        disable_branch_math=args.disable_branch_math,
    )

    out = output_path if output_path else output_root / f"{args.model}_{args.dataset_key}_forward_maxturn{args.max_cur_turn}_parallel_infer.json"

    def run_one(dp):
        agent = None
        try:
            with sample_budget_scope(str(dp["id"])):
                agent = progco.ProgCoAgent.__new__(progco.ProgCoAgent)
                agent.dataset = "MATH"
                agent.model = args.model
                agent.max_cur_turn = args.max_cur_turn
                agent.temperature = args.temperature
                sample_python_tool_cache = {}
                agent.llm_code_executor = PythonToolLLMCodeExecutor(
                    model=args.model,
                    temperature=args.temperature,
                    enable_branch_math=not args.disable_branch_math,
                    branch_math_model=args.branch_math_model or args.model,
                    successful_tool_flow_cache=(
                        sample_python_tool_cache
                    ),
                )
                agent.llm_code_executor.sample_id = dp["id"]
                agent.feedbacker = progco.Feebacker(args.model, args.temperature)
                agent.logger = MemoryLogger(run_args)
                agent.first_response_data = first_map
                agent.inference({"id": dp["id"], "query": dp["query"], "sample": dp})
                return agent.logger.logs[0]
        except SampleBudgetExceeded as error:
            if agent is not None and getattr(agent, "logger", None) is not None and agent.logger.logs:
                log = agent.logger.logs[0]
            else:
                log = {"id": dp["id"], "query": dp["query"], "turns": []}
            log["stopped"] = True
            log["stop_reason"] = "sample_operation_budget_exceeded"
            log["sample_operation_budget"] = error.to_dict()
            return log
        except KeyboardInterrupt:
            raise
        except Exception as e:
            execute_content = ""
            if agent is not None and getattr(agent, "llm_code_executor", None) is not None:
                execute_content = getattr(agent.llm_code_executor, "execute_content", "") or ""
            empty_content = is_empty_content_error(e)

            log_runtime_event({
                "sample_id": dp["id"],
                "stage": "sample",
                "status": "skipped" if empty_content else "failed",
                "error": f"{type(e).__name__}: {str(e)}",
                "skip_reason": EMPTY_CONTENT_SKIP_REASON if empty_content else "",
            })

            if agent is not None and getattr(agent, "logger", None) is not None and agent.logger.logs:
                log = agent.logger.logs[0]
                log["failed"] = True
                log["error"] = f"{type(e).__name__}: {str(e)}"
                if empty_content:
                    log["skipped"] = True
                    log["skip_reason"] = EMPTY_CONTENT_SKIP_REASON
                    log["empty_content"] = True
                    log["final_response"] = ""
                    log["final_result"] = None
                log["failed_execute_content"] = execute_content
                log["failed_python_tool_events"] = extract_python_tool_events(execute_content)
                log["failed_timing_events"] = extract_timing_events(execute_content)
                log["reasoning_trace_path"] = os.environ.get("SPQ_REASONING_TRACE_PATH", "")
                return log

            return build_failed_sample_log(dp, e, execute_content)

    logs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, dp) for dp in data]
        for i, fut in enumerate(as_completed(futures), 1):
            logs.append(fut.result())
            result = {"args": display_args(args, public_root), "result": None, "logs": sorted(logs, key=lambda x: x["id"])}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"done {i}/{len(futures)}", flush=True)

    result = {"args": display_args(args, public_root), "result": None, "logs": sorted(logs, key=lambda x: x["id"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
