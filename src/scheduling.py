from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Condition
from typing import TypeVar

DEFAULT_LIMITS = {
    "hf_download": 2,
    "split": 1,
    "docker_build": 1,
    "docker_push": 1,
    "docker_pull": 2,
    "extract_restore": 1,
}
DISK_HEADROOM = 1024**3

T = TypeVar("T")


def confirm(message: str) -> bool:
    print(message, file=sys.stderr, flush=True)
    if not sys.stdin.isatty():
        return False
    try:
        return input("Continue? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _docker_output(*args: str) -> str:
    return subprocess.check_output(
        ["docker", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def docker_storage() -> Path | None:
    try:
        endpoint = (
            os.environ.get("DOCKER_HOST")
            if not os.environ.get("DOCKER_CONTEXT")
            else None
        )
        if not endpoint:
            context = _docker_output("context", "show")
            endpoint = _docker_output(
                "context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}"
            )
        if not endpoint.startswith("unix://"):
            return None
        location = Path(_docker_output("info", "--format", "{{.DockerRootDir}}"))
        return location if location.is_absolute() else None
    except (OSError, subprocess.CalledProcessError):
        return None


class Storage:
    def __init__(self, path: Path | None, *, docker: bool = False) -> None:
        self.path = path
        self.docker = docker
        self.sudo = False

    def _local(self) -> tuple[int, int] | None:
        path = self.path
        if path is None:
            return None
        try:
            while True:
                try:
                    device = path.stat().st_dev
                    return device, shutil.disk_usage(path).free
                except FileNotFoundError:
                    if self.docker or path == path.parent:
                        return None
                    path = path.parent
        except OSError:
            return None

    def _elevated(self, *, authenticate: bool = False) -> tuple[int, int] | None:
        if self.path is None:
            return None
        prefix = ["sudo"] if authenticate else ["sudo", "-n"]
        try:
            device = subprocess.check_output(
                [*prefix, "stat", "-c", "%d", "--", str(self.path)], text=True
            )
            data = subprocess.check_output(
                [*prefix, "stat", "-f", "-c", "%a %S", "--", str(self.path)], text=True
            )
            blocks, size = map(int, data.split())
            return int(device), blocks * size
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None

    def measure(self) -> tuple[int, int] | None:
        result = self._local()
        return result if result is not None or not self.sudo else self._elevated()

    def authorize_probe(self) -> None:
        if (
            self.path
            and self._local() is None
            and shutil.which("sudo")
            and sys.stdin.isatty()
            and confirm(
                f"Docker storage {self.path} cannot be measured. Use sudo for read-only stat?"
            )
        ):
            self.sudo = self._elevated(authenticate=True) is not None


class Scheduler:
    """One task budget for the entire command, including filesystem reservations."""

    def __init__(
        self,
        limits: dict[str, int],
        storage: dict[str, Storage],
        *,
        overlap: bool,
        unknown: set[str],
    ) -> None:
        self.limits = limits
        self.storage = storage
        self.unknown = unknown
        self.overlap = overlap
        self.active: Counter[str] = Counter()
        self.reserved: Counter[int] = Counter()
        self.changed = Condition()
        self.generation = 0
        self.pool = ThreadPoolExecutor(max_workers=sum(limits.values()))

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def mark_change(self) -> None:
        with self.changed:
            self.generation += 1
            self.changed.notify_all()

    def submit(
        self, kind: str, estimates: dict[str, int], work: Callable[[], T]
    ) -> Future[T] | None:
        with self.changed:
            if self.active[kind] >= self.limits[kind] or (
                not self.overlap and any(self.active.values())
            ):
                return None
            demand: Counter[int] = Counter()
            free: dict[int, int] = {}
            for role, size in estimates.items():
                if not size:
                    continue
                result = self.storage[role].measure()
                if result is None:
                    if role in self.unknown:
                        continue
                    raise RuntimeError(
                        f"Disk capacity for {role} can no longer be verified"
                    )
                device, remaining = result
                demand[device] += size
                free[device] = min(remaining, free.get(device, remaining))
            if any(
                free[dev] < size + self.reserved[dev] + DISK_HEADROOM
                for dev, size in demand.items()
            ):
                return None
            self.active[kind] += 1
            self.reserved.update(demand)
            try:
                future = self.pool.submit(work)
            except BaseException:
                self.active[kind] -= 1
                self.reserved.subtract(demand)
                raise

            def release(_: Future[T]) -> None:
                with self.changed:
                    self.active[kind] -= 1
                    self.reserved.subtract(demand)
                    self.mark_change()

            future.add_done_callback(release)
            return future

    def wait_for_change(self, generation: int) -> None:
        with self.changed:
            if generation != self.generation:
                return
            if not any(self.active.values()):
                raise RuntimeError(
                    "No task can proceed within the disk budget; checkpoint retained for resume"
                )
            self.changed.wait_for(lambda: self.generation != generation)
