"""Independent, single-worker delivery of verified restored files over SSH."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import stat
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.delivery_remote import META, SUFFIX
from src.files import path_matches
from src.run_logging import logger
from src.state import expected_restored_path
from src.transfer import normalize_destination, ssh_options

if TYPE_CHECKING:
    from src.pipeline import ModelPipeline, Record

FileStamp = tuple[int, int, int, int, int]


def file_stamp(path: Path) -> FileStamp:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular local file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def destination_id(args: argparse.Namespace) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                args.ssh_host,
                normalize_destination(args.destination_path),
                args.ssh_option,
            ]
        ).encode()
    ).hexdigest()


def previously_delivered(model: ModelPipeline, raw: Record) -> bool:
    if not model.delivering or getattr(model.args, "keep_local", False):
        return False
    return (
        model.state.get("delivery_receipts", {})
        .get(destination_id(model.args), {})
        .get(raw["filename"])
        == raw["sha256"]
        and not model.output_path(raw).exists()
    )


def validate_delivery(model: ModelPipeline) -> None:
    names = {Path(raw["filename"]) for raw in model.state["raw"]}
    if model.name.startswith(META) or model.name.endswith(SUFFIX):
        raise ValueError(f"Reserved delivery model name: {model.name}")
    for path in names:
        if (
            path.parts[0] == META
            or "\x00" in str(path)
            or any(parent in names for parent in path.parents)
        ):
            raise ValueError(f"Conflicting or reserved delivery path: {path}")
    ssh_options(model.args.ssh_host, model.args.ssh_option)
    normalize_destination(model.args.destination_path)


def _transient(exc: subprocess.CalledProcessError) -> bool:
    error = (exc.stderr or "").lower()
    if "delivery receiver:" in error or any(
        text in error
        for text in (
            "permission denied",
            "host key verification failed",
            "remote host identification has changed",
            "could not resolve hostname",
            "bad configuration option",
        )
    ):
        return False
    return exc.returncode in {10, 30, 35} or any(
        text in error
        for text in (
            "connection timed out",
            "connection refused",
            "connection reset",
            "broken pipe",
            "connection closed",
            "operation timed out",
            "no route to host",
            "server not responding",
        )
    )


def run_with_retry(command: list[str], data: bytes | None = None) -> str:
    for attempt in range(3):
        try:
            result = subprocess.run(
                command,
                input=data or b"",
                capture_output=True,
                check=True,
                timeout=6 * 3600,
            )
            return result.stdout.decode()
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            failure = subprocess.CalledProcessError(
                exc.returncode, command[0], stderr=stderr
            )
            if attempt == 2 or not _transient(failure):
                raise RuntimeError(
                    f"{command[0]} exited {exc.returncode}: {stderr}"
                ) from failure
            logger.warning(
                "Transient %s failure; retrying (%d/2): %s",
                command[0],
                attempt + 1,
                stderr,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{command[0]} exceeded the delivery command timeout"
            ) from None
        time.sleep(2**attempt)
    raise AssertionError("Unreachable retry state")


class Receiver:
    def __init__(self, model: ModelPipeline) -> None:
        args = model.args
        self.name = model.name
        self.root = normalize_destination(args.destination_path)
        self.host = args.ssh_host
        self.transport = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            "RemoteCommand=none",
            "-o",
            "SessionType=default",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ForkAfterAuthentication=no",
            *ssh_options(self.host, args.ssh_option),
        ]
        self.manifest = json.dumps(
            {
                raw["filename"]: [raw["size_bytes"], raw["sha256"]]
                for raw in model.state["raw"]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.identity = hashlib.sha256(self.manifest).hexdigest()
        self.source = (
            Path(__file__).with_name("delivery_remote.py").read_text(encoding="utf-8")
        )

    def command(self, op: str, *args: str) -> str:
        return shlex.join(
            [
                "python3",
                "-c",
                self.source,
                op,
                self.root,
                self.name,
                self.identity,
                *args,
            ]
        )

    def inspect(self) -> tuple[bool, set[str]]:
        if not shutil.which("rsync") or not shutil.which("ssh"):
            raise RuntimeError(
                "deliver_llm requires local rsync and the OpenSSH client"
            )
        stdout = run_with_retry(
            [
                *self.transport,
                self.host,
                "command -v rsync >/dev/null || { echo 'deliver_llm requires remote rsync' >&2; exit 1; }; "
                + self.command("inspect"),
            ],
            self.manifest,
        )
        result = json.loads(stdout)
        return result["published"], set(result["verified"])

    def file(self, path: Path, raw: Record, verified: bool) -> FileStamp | None:
        root = path.parents[len(Path(raw["filename"]).parts) - 1]
        if path.is_symlink() or any(
            parent.is_symlink()
            for parent in path.parents
            if parent.is_relative_to(root)
        ):
            raise ValueError(f"Symlinks are not supported in restored payload: {path}")
        stamp = file_stamp(path)
        if verified:
            # Never remove unexpected local content when the destination was already correct.
            return (
                stamp if path_matches(path, raw["size_bytes"], raw["sha256"]) else None
            )
        if stamp[2] != raw["size_bytes"]:
            raise ValueError(f"Local file does not match the manifest: {path}")
        run_with_retry(
            [
                "rsync",
                "--times",
                "--inplace",
                "--no-whole-file",
                "--ignore-times",
                "--no-protect-args",
                "--timeout=60",
                "-e",
                shlex.join(self.transport),
                "--rsync-path",
                self.command("rsync", raw["filename"]),
                "--",
                str(path.absolute()),
                f"{self.host}:download-llm-partial",
            ]
        )
        run_with_retry(
            [*self.transport, self.host, self.command("verify", raw["filename"])],
            self.manifest,
        )
        return stamp

    def publish(self) -> None:
        run_with_retry(
            [*self.transport, self.host, self.command("publish")], self.manifest
        )


@dataclass
class Target:
    model: ModelPipeline
    receiver: Receiver
    inspected: bool = False
    published: bool = False
    verified: set[str] = field(default_factory=set)
    handled: set[str] = field(default_factory=set)
    failure: Exception | None = None


class Delivery:
    """Dispatch independently of restore limits; commit receipts on the coordinator."""

    def __init__(self, notify: Callable[[], None]) -> None:
        self.notify = notify
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.targets: dict[str, Target] = {}
        self.failures: dict[str, Exception] = {}
        self.running: tuple[Target, str, Record | None, Future[Any], float] | None = (
            None
        )
        self.sent_bytes = 0
        self.send_seconds = 0.0

    def close(self) -> None:
        self.worker.shutdown(wait=True)

    def add(self, model: ModelPipeline) -> None:
        try:
            self.targets[model.name] = Target(model, Receiver(model))
        except (OSError, ValueError) as exc:
            self.failures[model.name] = exc
            logger.error("Delivery setup failed [%s]: %s", model.name, exc)

    def _submit(
        self, target: Target, op: str, raw: Record | None, work: Callable[[], Any]
    ) -> None:
        started = time.monotonic()
        future = self.worker.submit(work)
        self.running = target, op, raw, future, started
        future.add_done_callback(lambda _: self.notify())

    def collect(self) -> bool:
        if self.running is None:
            return False
        target, op, raw, future, started = self.running
        if not future.done():
            return False
        self.running = None
        model = target.model
        try:
            result = future.result()
            if op == "inspect":
                target.published, target.verified = result
                target.inspected = True
            elif op == "publish":
                target.published = True
            elif raw is not None:
                name = raw["filename"]
                if name not in target.verified:
                    self.sent_bytes += raw["size_bytes"]
                    self.send_seconds += time.monotonic() - started
                target.verified.add(name)
                target.handled.add(name)
                logger.info("Delivered [%s] %s (SHA-256 verified)", model.name, name)
                if result is not None and not model.args.keep_local:
                    local = model.directory / expected_restored_path(name)
                    if file_stamp(local) != result:
                        raise RuntimeError(
                            f"Retaining file changed during delivery: {local}"
                        )
                    receipts = model.state.setdefault(
                        "delivery_receipts", {}
                    ).setdefault(destination_id(model.args), {})
                    receipts[name] = raw["sha256"]
                    model.save()  # A crash after unlink can still resume without restoring again.
                    local.unlink()
                    for record in model.state["restore"]:
                        if record["raw_filename"] == name:
                            record["deleted"] = True
                            break
                    model.save()
        except Exception as exc:  # noqa: BLE001 - isolate delivery from pull/restore
            target.failure = self.failures[model.name] = exc
            logger.error("Delivery failed [%s] (%s): %s", model.name, op, exc)
        return True

    def submit_next(self) -> bool:
        if self.running is not None:
            return False
        candidates: list[tuple[int, Target, Record, Path]] = []
        publish: list[Target] = []
        for target in self.targets.values():
            if target.failure is not None:
                continue
            if not target.inspected:
                self._submit(target, "inspect", None, target.receiver.inspect)
                return True
            model = target.model
            raw_list = model.state["raw"]
            for raw in raw_list:
                name = raw["filename"]
                if name in target.handled or (
                    name in target.verified and model.args.keep_local
                ):
                    continue
                if model.restoring and name not in model.restored:
                    continue
                path = model.output_path(raw)
                if path.is_file():
                    candidates.append((raw["size_bytes"], target, raw, path))
            if len(target.verified) == len(raw_list) and not target.published:
                publish.append(target)
        if candidates:
            _, target, raw, path = max(candidates, key=lambda item: item[0])
            verified = raw["filename"] in target.verified
            self._submit(
                target, "file", raw, lambda: target.receiver.file(path, raw, verified)
            )
            return True
        if publish:
            target = publish[0]
            self._submit(target, "publish", None, target.receiver.publish)
            return True
        return False

    def finish(self) -> None:
        for name, target in self.targets.items():
            if not target.published and name not in self.failures:
                missing = len(target.model.state["raw"]) - len(target.verified)
                self.failures[name] = RuntimeError(
                    f"{missing} file(s) not verified at destination; rerun delivery after restoring missing local files"
                )
            if target.model.restoring:
                restored = len(target.model.restored)
                total = len(target.model.state["raw"])
                logger.info(
                    "Restore [%s]: %s (%d/%d files)",
                    name,
                    "done"
                    if target.model.failure is None and restored == total
                    else "incomplete",
                    restored,
                    total,
                )
            if target.published and name not in self.failures:
                logger.info("Delivery [%s]: done", name)
        for name, exc in self.failures.items():
            logger.error("Delivery [%s]: incomplete: %s", name, exc)
