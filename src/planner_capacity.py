from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class ResourceCapacity:
    cpu_threads: float | None
    available_memory_bytes: int | None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _cgroups() -> tuple[Path, ...]:
    for line in _read(Path("/proc/self/cgroup")).splitlines():
        if line.startswith("0::/"):
            path = _CGROUP_ROOT / line[4:]
            if ".." in path.parts:
                break
            return (
                path,
                *(
                    parent
                    for parent in path.parents
                    if parent.is_relative_to(_CGROUP_ROOT)
                ),
            )
    return ()


def measure_resources(*, docker: bool) -> ResourceCapacity:
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count()
    cpu_limits = [float(cpus)] if cpus else []
    memory_limits: list[int] = []
    for line in _read(Path("/proc/meminfo")).splitlines():
        if line.startswith("MemAvailable:"):
            value = line.split()[1:2]
            if value and value[0].isdigit():
                memory_limits.append(int(value[0]) * 1024)
            break
    for group in _cgroups():
        cpu = _read(group / "cpu.max").split()
        if len(cpu) == 2 and all(value.isdigit() and int(value) > 0 for value in cpu):
            cpu_limits.append(int(cpu[0]) / int(cpu[1]))
        maximum, usage = _read(group / "memory.max"), _read(group / "memory.current")
        if maximum.isdigit() and usage.isdigit():
            memory_limits.append(max(0, int(maximum) - int(usage)))

    if docker:
        try:
            info = subprocess.check_output(
                ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            docker_cpus, docker_memory = map(int, info.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        else:
            if docker_cpus > 0:
                cpu_limits.append(float(docker_cpus))
            # Engine MemTotal is an upper bound, not an available-memory reading.
            if memory_limits and docker_memory > 0:
                memory_limits.append(docker_memory)

    return ResourceCapacity(
        min(cpu_limits, default=None), min(memory_limits, default=None)
    )
