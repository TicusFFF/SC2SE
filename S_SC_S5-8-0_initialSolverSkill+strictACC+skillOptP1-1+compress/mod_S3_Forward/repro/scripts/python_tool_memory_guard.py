"""Emergency protection for active kernels in one PythonTool pool only."""
import json
import logging
import os
import time
from pathlib import Path

import psutil


class PythonToolMemoryPressure(RuntimeError):
    pass


def memory_snapshot():
    # Same effective-memory calculation as PythonTool's existing admission check.
    if os.name == "nt":
        memory = psutil.virtual_memory()
        return memory.available, memory.total
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


def _try_lock(handle):
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _active_kernels(directory):
    candidates = []
    for path in directory.glob("active_*.json"):
        try:
            record = json.loads(path.read_text())
            process = psutil.Process(record["pid"])
            if (process.create_time() != record["created"]
                    or process.ppid() != record["server_pid"]
                    or process.status() == psutil.STATUS_ZOMBIE):
                continue  # Old records/PID reuse must never select another process.
            tree = process.children(recursive=True) + [process]
            rss = sum(child.memory_info().rss for child in tree)
            candidates.append((rss, process.pid, tree))
        except (OSError, ValueError, KeyError, psutil.Error):
            continue
    return candidates


class RunningKernelMemoryGuard:
    def __init__(self, kernel):
        # Enable the running-kernel guard whenever admission protection is enabled.
        self.enabled = float(os.environ.get("SPQ_PYTHON_TOOL_MEMORY_RESERVE_RATIO", "0.25")) > 0
        self.kernel = kernel
        self.path = None
        self.next_check = 0.0
        self.last_rss = None
        if self.enabled:
            directory = Path(os.environ.get("SPQ_PYTHON_TOOL_LOCK_DIR") or
                             (Path(os.environ.get("TEMP") or os.getcwd()) / "forward_python_tool_locks"))
            self.directory = directory / "active_kernels"
            self.directory.mkdir(parents=True, exist_ok=True)
            process = psutil.Process(kernel.pid)
            assert process.ppid() == os.getpid(), "Only this server's child kernel can be registered."
            self.path = self.directory / f"active_{os.getpid()}.json"
            temporary = self.path.with_suffix(".pending")
            temporary.write_text(json.dumps({
                "pid": process.pid, "created": process.create_time(), "server_pid": os.getpid(),
            }))
            os.replace(temporary, self.path)

    def check(self):
        if not self.enabled or time.monotonic() < self.next_check:
            return
        self.next_check = time.monotonic() + 1.0
        try:
            process = psutil.Process(self.kernel.pid)
            rss = sum(p.memory_info().rss for p in process.children(recursive=True) + [process])
            growth = None if self.last_rss is None else rss - self.last_rss
            self.last_rss = rss
            available, total = memory_snapshot()
            # Cap the emergency reserve on large-memory hosts; keep 10% on smaller ones.
            emergency_reserve = min(int(total * 0.10), 50 * 1024 ** 3)
            if available >= emergency_reserve:
                return  # Admission alone pauses new work; active work is not stopped here.
            # Serialize emergency decisions across servers and re-read memory after locking.
            with (self.directory / "emergency.lock").open("a+b") as lock:
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                if not _try_lock(lock):
                    return
                available, total = memory_snapshot()
                emergency_reserve = min(int(total * 0.10), 50 * 1024 ** 3)
                if available >= emergency_reserve:
                    return
                candidates = _active_kernels(self.directory)
                if not candidates:
                    return
                largest_rss, largest_pid, tree = max(candidates, key=lambda item: (item[0], item[1]))
                if largest_pid != self.kernel.pid or self.kernel.poll() is not None:
                    return  # Each server may stop only its own active kernel.
                # Stop the kernel first, then its already identified children; never the server.
                for child in reversed(tree):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(tree, timeout=1.0)
                print("[PythonTool memory protection] " + json.dumps({
                    "status": "terminated", "kernel_pid": largest_pid,
                    "rss_bytes": largest_rss, "growth_bytes": growth,
                    "available_bytes": available, "total_bytes": total,
                    "emergency_ratio": 0.10,
                    "emergency_reserve_bytes": emergency_reserve,
                }), flush=True)
            raise PythonToolMemoryPressure(
                "RuntimeError: PythonTool execution stopped by memory protection: "
                "less than min(10% of effective memory, 50 GiB) remained and this was the largest active kernel. "
                "Its kernel state was reset. Use a more memory-efficient calculation."
            )
        except PythonToolMemoryPressure:
            raise
        except (OSError, ValueError, KeyError, psutil.Error):
            # Never choose a victim from incomplete/unverifiable process information.
            logging.warning("PythonTool emergency memory check failed", exc_info=True)

    def close(self):
        if self.path is not None:
            self.path.unlink(missing_ok=True)
