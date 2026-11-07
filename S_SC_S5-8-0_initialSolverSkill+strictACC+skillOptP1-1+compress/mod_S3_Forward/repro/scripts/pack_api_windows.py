# -*- coding: utf-8 -*-
import logging
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request
from gevent import pywsgi
from python_tool_memory_guard import PythonToolMemoryPressure, RunningKernelMemoryGuard


CODE_INTERPRETER_DIR = Path(__file__).resolve().parents[2] / "upstream" / "progco" / "code_interpreter"
os.environ.setdefault("SPQ_PYTHON_TOOL_MEMORY_RESERVE_RATIO", "0.25")
os.chdir(CODE_INTERPRETER_DIR)
sys.path.insert(0, str(CODE_INTERPRETER_DIR))

import code_interpreter as ci


START_CODE_NO_SIGNAL = """
def input(*args, **kwargs):
    raise NotImplementedError('Python input() function is disabled.')

import os
if 'upload_file' not in os.getcwd():
    os.chdir("./upload_file/")

import math
import re
import json
"""


def _code_interpreter_no_alarm(code: str, timeout, clear=False):
    if not code.strip():
        return ""
    if clear:
        code = "get_ipython().run_line_magic('reset', '-f')\n" + START_CODE_NO_SIGNAL + code

    pid = ci.os.getpid()
    if pid not in ci._KERNEL_CLIENTS:
        ci._KERNEL_CLIENTS[pid] = ci.start_kernel(pid)
        _code_interpreter_no_alarm(START_CODE_NO_SIGNAL, timeout=None)

    kc = ci._KERNEL_CLIENTS[pid]
    kc.wait_for_ready()
    guard = RunningKernelMemoryGuard(ci._KERNEL_PROCESSES[pid])
    try:
        kc.execute(code)
    except BaseException:
        guard.close()
        raise
    result = ""
    image_idx = 0
    timed_out = False
    memory_stopped = False
    deadline = (
        None
        if timeout is None
        else time.monotonic() + max(0.0, float(timeout))
    )

    while True:
        text = ""
        image = ""
        finished = False
        msg_type = "error"
        try:
            guard.check()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise queue.Empty
            if guard.enabled:
                remaining = 1.0 if remaining is None else min(1.0, remaining)
            msg = kc.get_iopub_msg(timeout=remaining)
            msg_type = msg["msg_type"]
            if msg_type == "status":
                if msg["content"].get("execution_state") == "idle":
                    finished = True
            elif msg_type == "execute_result":
                text = msg["content"]["data"].get("text/plain", "")
                if "image/png" in msg["content"]["data"]:
                    image_url = ci.publish_image_to_local(msg["content"]["data"]["image/png"])
                    image_idx += 1
                    image = "![fig-%03d](%s)" % (image_idx, image_url)
            elif msg_type == "display_data":
                if "image/png" in msg["content"]["data"]:
                    image_url = ci.publish_image_to_local(msg["content"]["data"]["image/png"])
                    image_idx += 1
                    image = "![fig-%03d](%s)" % (image_idx, image_url)
                else:
                    text = msg["content"]["data"].get("text/plain", "")
            elif msg_type == "stream":
                msg_type = msg["content"]["name"]
                text = msg["content"]["text"]
            elif msg_type == "error":
                text = ci.escape_ansi("\n".join(msg["content"]["traceback"]))
        except PythonToolMemoryPressure as error:
            text = str(error)
            finished = True
            memory_stopped = True
        except queue.Empty:
            if guard.enabled and (deadline is None or time.monotonic() < deadline):
                continue  # One-second memory checks do not shorten the original execution timeout.
            text = f"Timeout. No response after {timeout} seconds."
            finished = True
            timed_out = True
        except Exception:
            text = "The code interpreter encountered an unexpected error."
            logging.warning("".join(traceback.format_exception(*sys.exc_info())))
            finished = True

        if text:
            result += f"\n\n{msg_type}:\n\n```\n{text}\n```"
        if image:
            result += f"\n\n{image}"
        if finished:
            break

    guard.close()
    if timed_out or memory_stopped:
        ci.stop_kernel(pid)

    return result.lstrip("\n")


ci.START_CODE = START_CODE_NO_SIGNAL
ci._code_interpreter = _code_interpreter_no_alarm

app = Flask(__name__)
CODE_LOCK = threading.Lock()


@app.route("/api/code_interpreter", methods=["POST"])
def api_code_interpreter():
    data = request.json
    if not CODE_LOCK.acquire(blocking=False):
        return jsonify({"busy": True, "result": "PythonTool server busy"}), 429
    try:
        execution_timeout = int(data.get("execution_timeout_seconds", 60))
        result = ci.code_interpreter(data.get("code"), timeout=execution_timeout)
    except Exception:
        result = "error:\n" + traceback.format_exc()
    finally:
        CODE_LOCK.release()
    return jsonify({"busy": False, "result": result})


if __name__ == "__main__":
    port = int(os.getenv("BRANCHMATH_CODE_INTERPRETER_PORT", "4200"))
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    server = pywsgi.WSGIServer(("127.0.0.1", port), app)
    server.serve_forever()
