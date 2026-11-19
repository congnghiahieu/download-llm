from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Condition
from typing import TypeVar

from src.run_logging import log_file

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
Device = int | str


def storage_constraints(role: str, available: Collection[str]) -> tuple[str, ...]:
    host = f"{role}_host"
    return (role, host) if host in available else (role,)


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
        ["docker", *args], text=True, stderr=subprocess.PIPE
    ).strip()


@dataclass(frozen=True)
class DockerDesktopRoot:
    path: Path


def docker_storage(*, include_desktop: bool = False) -> Path | DockerDesktopRoot | None:
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
            log_file(
                logging.INFO,
                "Docker storage is not measurable locally: endpoint scheme=%r",
                endpoint.split(":", 1)[0],
            )
            return None
        daemon_os, _, root = _docker_output(
            "info", "--format", "{{.OperatingSystem}}\n{{.DockerRootDir}}"
        ).partition("\n")
        log_file(
            logging.INFO, "Docker daemon storage [os=%r, root=%r]", daemon_os, root
        )
        location = Path(root)
        if "docker desktop" in daemon_os.lower():
            if include_desktop and location.is_absolute():
                return DockerDesktopRoot(location)
            log_file(
                logging.INFO,
                "Docker Desktop storage belongs to its daemon environment, not the CLI filesystem",
            )
            return None
        return location if location.is_absolute() else None
    except (OSError, subprocess.CalledProcessError) as exc:
        log_file(
            logging.WARNING,
            "Docker storage probe failed: %s %s",
            exc,
            getattr(exc, "stderr", ""),
        )
        return None


class Storage:
    def __init__(
        self,
        path: Path | None,
        *,
        docker: bool = False,
        unavailable_reason: str | None = None,
    ) -> None:
        self.path = path
        self.docker = docker
        self.sudo = False
        self.measurement_error = unavailable_reason

    def _local(self) -> tuple[int, int] | None:
        path = self.path
        if path is None:
            self.measurement_error = (
                self.measurement_error or "storage path not available locally"
            )
            return None
        try:
            while True:
                try:
                    device = path.stat().st_dev
                    free = shutil.disk_usage(path).free
                    self.measurement_error = None
                    return device, free
                except FileNotFoundError as exc:
                    if self.docker or path == path.parent:
                        self.measurement_error = str(exc)
                        return None
                    path = path.parent
        except OSError as exc:
            self.measurement_error = str(exc)
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
            self.measurement_error = None
            return int(device), blocks * size
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            self.measurement_error = f"sudo storage probe: {exc}"
            return None

    def measure(self) -> tuple[Device, int] | None:
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
        self.reserved: Counter[Device] = Counter()
        self.changed = Condition()
        self.generation = 0
        self.last_disk_block: dict[str, int | str] | None = None
        self.pool = ThreadPoolExecutor(max_workers=sum(limits.values()))

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def mark_change(self) -> None:
        with self.changed:
            self.generation += 1
            self.changed.notify_all()

    def disk_snapshot(self) -> dict[str, dict[str, object]]:
        with self.changed:
            snapshot: dict[str, dict[str, object]] = {}
            for role, store in self.storage.items():
                result = store.measure()
                snapshot[role] = {"path": str(store.path) if store.path else None}
                if result is None:
                    snapshot[role].update(
                        free_bytes=None, reason=store.measurement_error
                    )
                else:
                    device, available = result
                    snapshot[role].update(
                        device=device,
                        free_bytes=available,
                        reserved_bytes=self.reserved[device],
                    )
            return snapshot

    def submit(
        self, kind: str, estimates: dict[str, int], work: Callable[[], T]
    ) -> Future[T] | None:
        with self.changed:
            if self.active[kind] >= self.limits[kind] or (
                not self.overlap and any(self.active.values())
            ):
                return None
            demand: Counter[Device] = Counter()
            free: dict[Device, int] = {}
            for role, size in estimates.items():
                if not size:
                    continue
                charged: set[Device] = set()
                for constraint in storage_constraints(role, self.storage):
                    store = self.storage[constraint]
                    result = store.measure()
                    if result is None:
                        if constraint in self.unknown:
                            continue
                        log_file(
                            logging.ERROR,
                            "Storage measurement lost [role=%s, path=%s]: %s",
                            constraint,
                            store.path,
                            store.measurement_error,
                        )
                        raise RuntimeError(
                            f"Disk capacity for {constraint} can no longer be verified"
                        )
                    device, remaining = result
                    if device not in charged:
                        demand[device] += size
                        charged.add(device)
                    free[device] = min(remaining, free.get(device, remaining))
            for dev, size in demand.items():
                if free[dev] < size + self.reserved[dev] + DISK_HEADROOM:
                    self.last_disk_block = {
                        "task": kind,
                        "device": dev,
                        "free_bytes": free[dev],
                        "requested_bytes": size,
                        "reserved_bytes": self.reserved[dev],
                        "headroom_bytes": DISK_HEADROOM,
                    }
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

    def wait_for_change(
        self, generation: int, *, external: bool = False, timeout: float | None = None
    ) -> None:
        with self.changed:
            if generation != self.generation:
                return
            if not external and not any(self.active.values()):
                log_file(
                    logging.ERROR,
                    "Scheduler stalled [last_disk_block=%r, storage=%r]",
                    self.last_disk_block,
                    self.disk_snapshot(),
                )
                raise RuntimeError(
                    "No task can proceed within the disk budget; checkpoint retained for resume"
                )
            self.changed.wait_for(lambda: self.generation != generation, timeout=timeout)
