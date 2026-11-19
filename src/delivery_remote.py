"""Receiver for deliver_llm, sent as source over SSH (no remote install)."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

META = ".download-llm-delivery"
SUFFIX = ".transferring"


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def directory(path: Path) -> None:
    path.mkdir(exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Not a real directory: {path}")


def payload(root: Path, name: str, *, create: bool = False) -> Path:
    path = root
    for folder in Path(name).parts[:-1]:
        path /= folder
        if create:
            directory(path)
        elif path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Conflicting payload directory: {path}")
    return root / name


def matches(path: Path, size: int, digest: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular payload file: {path}")
    if info.st_size != size:
        return False
    with path.open("rb") as file:
        hashed = hashlib.sha256()
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest() == digest


def sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_directory(source: Path, target: Path) -> None:
    # Linux renameat2 with RENAME_NOREPLACE also protects against writers that
    # do not take our receiver lock. Plain rename can replace an empty directory.
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def sync_staged_file(path: Path, stage: Path) -> None:
    sync(path)
    for parent in path.parents:
        sync(parent)
        if parent == stage:
            break


def expected_tree(root: Path, names: set[str], *, staging: bool) -> None:
    parents = {p.as_posix() for name in names for p in Path(name).parents}
    for folder, dirs, files in os.walk(root):
        path = Path(folder)
        if staging and path == root and META in dirs:
            dirs.remove(META)
        for dirname in dirs:
            child = path / dirname
            if child.is_symlink() or child.relative_to(root).as_posix() not in parents:
                raise ValueError(f"Unexpected destination directory: {child}")
        for filename in files:
            child = path / filename
            if child.relative_to(root).as_posix() not in names:
                raise ValueError(f"Unexpected destination file: {child}")


def receive(op: str, root: Path, name: str, identity: str, args: list[str]) -> None:
    incoming = sys.stdin.buffer.read() if op != "rsync" else b""
    if op != "rsync" and fingerprint(incoming) != identity:
        raise ValueError("Manifest identity mismatch")
    root.mkdir(parents=True, exist_ok=True)
    locks = root / (META + "-locks")
    directory(locks)
    with (locks / fingerprint(name.encode())).open("a") as lock:
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError(
                        f"Another delivery is using the destination for {name}"
                    ) from None
                time.sleep(0.25)
        final, stage = root / name, root / (name + SUFFIX)
        finished = final.exists() or final.is_symlink()
        if finished:
            if final.is_symlink() or not final.is_dir():
                raise ValueError(f"Conflicting destination: {final}")
            if op == "rsync":
                raise ValueError(f"Destination published during transfer: {final}")
            manifest = json.loads(incoming)
            if op == "verify":
                key = args[0]
                if not matches(payload(final, key), *manifest[key]):
                    raise ValueError(
                        f"Published payload does not match manifest: {key}"
                    )
                return
            expected_tree(final, set(manifest), staging=False)
            if not all(
                matches(payload(final, key), *values)
                for key, values in manifest.items()
            ):
                raise ValueError(
                    f"Existing destination does not match manifest: {final}"
                )
            print(json.dumps({"published": True, "verified": list(manifest)}))
            return
        meta = stage / META
        saved_meta = locks / (fingerprint(name.encode()) + ".publishing")
        if stage.is_symlink() or meta.is_symlink() or saved_meta.is_symlink():
            raise ValueError(f"Invalid staging directory: {stage}")
        if saved_meta.exists() and stage.is_dir() and not meta.exists():
            saved_meta.rename(meta)
        manifest_file = meta / "manifest.json"
        if op == "inspect":
            directory(stage)
            directory(meta)
            if not manifest_file.exists():
                if any(p != meta for p in stage.iterdir()) or any(
                    p.name != "manifest.tmp" for p in meta.iterdir()
                ):
                    raise ValueError(f"Staging directory has no manifest: {stage}")
                temporary = meta / "manifest.tmp"
                if temporary.is_symlink():
                    raise ValueError(f"Conflicting manifest path: {temporary}")
                temporary.write_bytes(incoming)
                sync(temporary)
                temporary.replace(manifest_file)
                sync(meta)
                sync(stage)
                sync(root)
        if manifest_file.is_symlink():
            raise ValueError(f"Invalid staging directory: {stage}")
        data = manifest_file.read_bytes()
        if fingerprint(data) != identity:
            raise ValueError(
                f"Staging directory belongs to a different manifest: {stage}"
            )
        manifest = json.loads(data)
        if op == "inspect":
            expected_tree(stage, set(manifest), staging=True)
            ready = []
            for key, values in manifest.items():
                path = payload(stage, key)
                if matches(path, *values):
                    sync_staged_file(path, stage)
                    ready.append(key)
            print(json.dumps({"published": False, "verified": ready}))
        elif op in {"rsync", "verify"}:
            key, *command = args
            size, digest = manifest[key]
            partial = meta / (fingerprint(key.encode()) + ".partial")
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise ValueError(f"Invalid partial file: {partial}")
            if op == "rsync":
                # OpenSSH appends the rsync server arguments. Only our partial file
                # may be written by that server.
                if (
                    not command
                    or command[0] != "--server"
                    or command[-1] != "download-llm-partial"
                ):
                    raise ValueError("Unexpected rsync server arguments")
                sys.exit(
                    subprocess.run(
                        ["rsync", *command[:-1], str(partial)], check=False
                    ).returncode
                )
            target = payload(stage, key, create=True)
            if not partial.exists() and matches(target, size, digest):
                sync_staged_file(target, stage)
                return  # A previous verify may have committed before SSH disconnected.
            if not matches(partial, size, digest):
                partial.unlink(missing_ok=True)
                raise ValueError(f"Transferred file failed SHA-256 verification: {key}")
            partial.replace(target)
            # Persist the file and any new ancestor directories before allowing local cleanup.
            sync_staged_file(target, stage)
            sync(meta)
        elif op == "publish":
            expected_tree(stage, set(manifest), staging=True)
            if not all(
                matches(payload(stage, key), *values)
                for key, values in manifest.items()
            ):
                raise ValueError(f"Staged payload does not match manifest: {stage}")
            # Rename only the payload directory, not the partials or manifest. Hold
            # the receiver lock throughout, including the final existence check.
            if saved_meta.exists():
                raise ValueError(f"Unfinished publish metadata: {saved_meta}")
            meta.rename(saved_meta)
            try:
                sync(stage)
                sync(locks)
                if final.exists() or final.is_symlink():
                    raise ValueError(f"Conflicting destination: {final}")
                publish_directory(stage, final)
            except BaseException:
                saved_meta.rename(meta)
                raise
            sync(root)
            for path in saved_meta.iterdir():
                path.unlink()
            saved_meta.rmdir()
            sync(locks)


if __name__ == "__main__":
    try:
        operation, destination, model, manifest_id, *extra = sys.argv[1:]
        receive(operation, Path(destination), model, manifest_id, extra)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Delivery receiver: {exc}", file=sys.stderr)
        sys.exit(1)
