from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.constants import (
    EXTRACTED_DIRNAME,
    PARTS_DIRNAME,
    RAW_DIRNAME,
    RESTORED_DIRNAME,
    STATE_FILENAME,
)
from src.huggingface import _hf_cache_dir, _hf_cache_paths, snapshot_cache_lock
from src.naming import docker_pull_tag
from src.sizes import human_size
from src.state import (
    effective_base_dir,
    load_state,
    model_lock,
    resolve_model_alias,
    save_state,
)

CATEGORIES = ("raw", "parts", "images", "extracted", "restored", "hf-cache")
_LOCAL_DIRS = {
    "raw": Path(RAW_DIRNAME),
    "parts": Path(PARTS_DIRNAME),
    "extracted": Path(EXTRACTED_DIRNAME, PARTS_DIRNAME),
    "restored": Path(EXTRACTED_DIRNAME, RESTORED_DIRNAME),
}
_HUB_CACHE = Path(RAW_DIRNAME, ".cache", "huggingface")
_REF = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]*|@sha256:[0-9a-f]{64})"
)


@dataclass(frozen=True)
class Target:
    location: str
    size: int
    identity: str


@dataclass(frozen=True)
class Item:
    targets: tuple[Target, ...] = ()
    error: str | None = None

    @property
    def size(self) -> int:
        return sum({target.identity: target.size for target in self.targets}.values())


@dataclass
class CleanupPlan:
    directory: Path
    items: dict[str, Item]
    state: dict[str, Any] | None
    snapshot: tuple[str, str] | None
    prefix: str | None
    state_error: str | None


def _location(root: Path, target: Path) -> os.stat_result | None:
    if target == root:
        return root.lstat()
    current = root
    info = None
    for part in target.relative_to(root).parts:
        if part in ("..", "."):
            raise ValueError(f"Unsafe cleanup path: {target}")
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode) or current.is_mount():
            raise ValueError(f"Refusing symlink or mount point: {current}")
    return info


def _mount_points() -> set[Path]:
    # is_mount alone misses Linux bind mounts on the same filesystem.
    return {
        Path(
            re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), row.split()[4])
        )
        for row in Path("/proc/self/mountinfo").read_text().splitlines()
    }


def _measure_dir(directory: Path, device: int) -> int:
    def fail(exc: OSError) -> None:
        raise exc

    total = 0
    for root, dirs, files in os.walk(directory, onerror=fail):
        for name in dirs + files:
            path = Path(root) / name
            info = path.lstat()
            if info.st_dev != device or path.is_mount():
                raise ValueError(f"Refusing mount point: {path}")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _file_target(
    root: Path, path: Path, mounts: set[Path], *, directory: bool
) -> Target | None:
    if any(
        mount != root
        and mount.is_relative_to(root)
        and (path.is_relative_to(mount) or (directory and mount.is_relative_to(path)))
        for mount in mounts
    ):
        raise ValueError(f"Refusing mounted cleanup path: {path}")
    info = _location(root, path)
    if info is None:
        return None
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode):
        raise ValueError(f"Unexpected file type: {path}")
    size = _measure_dir(path, info.st_dev) if directory else info.st_size
    return Target(str(path), size, f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}")


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False
    )


def _docker_local() -> None:
    endpoint = (
        None if os.environ.get("DOCKER_CONTEXT") else os.environ.get("DOCKER_HOST")
    )
    if not endpoint:
        result = _docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or "Cannot identify Docker endpoint"
            )
        endpoint = result.stdout.strip()
    if not endpoint.startswith("unix://"):
        raise RuntimeError("Cleanup requires a local Docker socket (unix://)")


def _refs(entry: dict[str, Any], prefix: str | None) -> set[str]:
    refs = {entry.get("tag"), entry.get("pull_tag")}
    if entry.get("digest") or prefix:
        refs.add(docker_pull_tag(entry, prefix))
    present = {ref for ref in refs if ref}
    if not present or any(
        not isinstance(ref, str) or not _REF.fullmatch(ref) for ref in present
    ):
        raise ValueError("Invalid Docker reference in manifest")
    return present


def _missing_image(detail: str) -> bool:
    return detail.lower().startswith(
        (
            "error: no such image:",
            "error: no such object:",
            "error response from daemon: no such image:",
        )
    )


def _inspect_image(ref: str) -> Target | None:
    result = _docker("image", "inspect", "--format", "{{.Id}} {{.Size}}", "--", ref)
    if result.returncode:
        detail = result.stderr.strip()
        if _missing_image(detail):
            return None
        raise RuntimeError(detail or f"Cannot inspect {ref}")
    image_id, size = result.stdout.split()
    return Target(ref, int(size), image_id)


def _image_targets(state: dict[str, Any], prefix: str | None) -> tuple[Target, ...]:
    refs = {ref for entry in state["dockerfiles"] for ref in _refs(entry, prefix)}
    if refs:
        _docker_local()
    targets = [
        target for ref in sorted(refs) if (target := _inspect_image(ref)) is not None
    ]
    # Remove digest references before tags that might implicitly drop those digests.
    return tuple(
        sorted(
            targets, key=lambda target: ("@" not in target.location, target.location)
        )
    )


def _snapshot(state: dict[str, Any] | None) -> tuple[str, str] | None:
    snapshot = (state or {}).get("snapshot")
    repo = (state or {}).get("repo_id")
    if isinstance(snapshot, dict) and isinstance(repo, str):
        commit = snapshot.get("commit")
        if (
            re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
            and isinstance(commit, str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", commit)
        ):
            return repo, commit
    return None


def _cache_targets(
    directory: Path,
    state: dict[str, Any] | None,
    snapshot: tuple[str, str] | None,
    mounts: set[Path],
) -> Item:
    local = _file_target(directory, directory / _HUB_CACHE, mounts, directory=True)
    found = [local] if local else []
    if snapshot is None or state is None:
        return Item(
            tuple(found),
            "Shared HF cache cannot be identified without a pinned snapshot",
        )
    root = _hf_cache_dir().resolve()
    paths: set[Path] = set()
    for filename, _ in state["snapshot"]["files"]:
        paths.update(path.absolute() for path in _hf_cache_paths(*snapshot, filename))
    # A user may intentionally configure a cache root through a symlink.
    configured = _hf_cache_dir().absolute()
    for path in sorted(paths):
        if target := _file_target(
            root, root / path.relative_to(configured), mounts, directory=False
        ):
            found.append(target)
    return Item(tuple(found))


def _inventory(
    directory: Path, prefix: str | None, selected: set[str] | None
) -> CleanupPlan:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Not a regular model directory: {directory}")
    mounts = _mount_points()
    if directory in mounts:
        raise ValueError(f"Refusing mounted model directory: {directory}")
    state = None
    state_error = None
    try:
        if (directory / STATE_FILENAME).is_symlink():
            raise ValueError("State is a symlink")
        state = load_state(directory)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        state_error = f"Cannot read model state: {exc}"
    snapshot = _snapshot(state)
    cache_root = _hf_cache_dir().resolve()
    items = {}
    for category in CATEGORIES:
        if selected is not None and category not in selected:
            continue
        try:
            if category == "images":
                items[category] = (
                    Item(_image_targets(state, prefix))
                    if state is not None
                    else Item(error=state_error)
                )
            elif category == "hf-cache":
                items[category] = _cache_targets(directory, state, snapshot, mounts)
            else:
                path = directory / _LOCAL_DIRS[category]
                if cache_root.is_relative_to(path):
                    raise ValueError(
                        f"Artifact directory contains the configured shared HF cache: {cache_root}"
                    )
                target = _file_target(directory, path, mounts, directory=True)
                items[category] = Item((target,) if target else ())
        except (
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            KeyError,
        ) as exc:
            items[category] = Item(error=str(exc))
    return CleanupPlan(directory, items, state, snapshot, prefix, state_error)


def make_cleanup_plan(
    args: argparse.Namespace, selected: set[str] | None = None
) -> CleanupPlan:
    root = effective_base_dir(args)
    name = args.model_name
    try:
        name = resolve_model_alias(name, root)
    except ValueError as exc:
        direct = root / name
        if (
            str(exc).startswith("Ambiguous")
            or Path(name).name != name
            or name in (".", "..")
            or direct.is_symlink()
            or not any(
                (direct / marker).exists()
                for marker in (*_LOCAL_DIRS.values(), STATE_FILENAME)
            )
        ):
            raise
    return _inventory(root / name, getattr(args, "docker_pull_prefix", None), selected)


def render_cleanup(plan: CleanupPlan, selected: set[str] | None = None) -> None:
    print(f"Local cleanup: {plan.directory}")
    if plan.state_error:
        print(plan.state_error)
    for index, category in enumerate(CATEGORIES, 1):
        if category not in plan.items or (
            selected is not None and category not in selected
        ):
            continue
        item = plan.items[category]
        covered = tuple(
            target
            for target in item.targets
            if selected
            and category == "hf-cache"
            and "raw" in selected
            and Path(target.location).is_relative_to(plan.directory / RAW_DIRNAME)
        )
        separate = Item(
            tuple(target for target in item.targets if target not in covered)
        )
        status = (
            human_size(separate.size)
            if separate.targets
            else "included in raw"
            if covered
            else "not present"
            if not item.error
            else "unavailable"
        )
        print(f"  {index}. {category}: {status}")
        for target in item.targets:
            if target in covered:
                print(f"     {target.location} (included in raw)")
            else:
                print(f"     {target.location} ({human_size(target.size)})")
        if item.error:
            print(f"     {item.error}")
    if (selected is None or "raw" in selected) and (
        plan.directory / _HUB_CACHE
    ).is_dir():
        print("Removing raw also removes its nested huggingface_hub cache.")
    print(
        "Listed sizes are not guaranteed disk savings (shared layers and hard links)."
    )


def _missing_file(directory: Path, relative: Path, path: Path) -> bool:
    if path.is_absolute() or ".." in path.parts or not path.is_relative_to(relative):
        return False
    try:
        return _location(directory, directory / path) is None
    except (OSError, ValueError):
        return False


def _record_cleanup(
    plan: CleanupPlan, selected: set[str], failed_images: set[str]
) -> None:
    state = plan.state
    if state is None:
        return
    updates: list[tuple[dict[str, Any], str, bool]] = []
    for category, records, filename, stored in (
        ("raw", state["raw"], "filename", "path"),
        ("parts", state["parts"], "part_filename", "path"),
        ("extracted", state["parts"], "part_filename", ""),
        ("restored", state["restore"], "raw_filename", "restored_path"),
    ):
        if category in selected:
            for record in records:
                name = record.get(filename)
                if not isinstance(name, str):
                    continue
                relative = _LOCAL_DIRS[category]
                path = Path(record.get(stored) or relative / name)
                if _missing_file(plan.directory, relative, path):
                    field = "extracted" if category == "extracted" else "deleted"
                    updates.append((record, field, category != "extracted"))
    if "images" in selected and not plan.items["images"].error:
        for entry in state["dockerfiles"]:
            if not (_refs(entry, plan.prefix) & failed_images):
                updates.append((entry, "removed", True))
    changed = False
    for record, field, value in updates:
        if record.get(field) != value:
            record[field] = value
            changed = True
    if changed:
        save_state(plan.directory, state)


def _remove_target(plan: CleanupPlan, category: str, target: Target) -> bool:
    if category == "images":
        current = _inspect_image(target.location)
        if current is None:
            return False
        if current != target:
            raise RuntimeError("Docker reference changed after confirmation")
        result = _docker("image", "rm", "--no-prune", "--", target.location)
        if result.returncode:
            detail = result.stderr.strip()
            if _missing_image(detail):
                return False
            raise RuntimeError(detail or f"Cannot remove {target.location}")
        return True
    path = Path(target.location)
    root = (
        plan.directory
        if path.is_relative_to(plan.directory)
        else _hf_cache_dir().resolve()
    )
    if info := _location(root, path):
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    return False


def apply_cleanup(preview: CleanupPlan, selected: set[str]) -> int:
    if not selected or not selected.issubset(CATEGORIES):
        raise ValueError("Select cleanup categories explicitly")
    if preview.directory.is_symlink() or preview.directory in _mount_points():
        raise ValueError(f"Unsafe model directory: {preview.directory}")
    for category in sorted(selected):
        _require_inspected(preview, category)
    with ExitStack() as locks:
        locks.enter_context(model_lock(preview.directory))
        # The cache root is shared independently of the model directory.
        if (
            "hf-cache" in selected
            and preview.snapshot
            and any(
                target.location != str(preview.directory / _HUB_CACHE)
                for target in preview.items["hf-cache"].targets
            )
        ):
            cache_root = _hf_cache_dir().resolve()
            cache_path = _hf_cache_paths(*preview.snapshot, "placeholder")[0].parent
            _location(
                cache_root,
                cache_root
                / cache_path.absolute().relative_to(_hf_cache_dir().absolute()),
            )
            locks.enter_context(snapshot_cache_lock(*preview.snapshot, wait=False))
        plan = _inventory(preview.directory, preview.prefix, selected)
        if plan.snapshot != preview.snapshot or any(
            plan.items[category] != preview.items[category] for category in selected
        ):
            raise RuntimeError(
                "Local data changed since preview; inspect it again before cleanup"
            )
        for category in sorted(selected):
            _require_inspected(plan, category)
        # Record this guard before deleting legacy receive data, so an interrupted
        # cleanup cannot cause a later pull from an unpinned remote tag.
        if (
            plan.state is not None
            and not plan.state.get("cleanup_requires_pinned_pull")
            and any(not entry.get("digest") for entry in plan.state["dockerfiles"])
            and any(
                plan.items[category].targets
                for category in selected & {"images", "extracted", "restored"}
            )
        ):
            plan.state["cleanup_requires_pinned_pull"] = True
            save_state(plan.directory, plan.state)
        failed_images: set[str] = set()
        removed: set[str] = set()
        status = 0
        try:
            for category, item in plan.items.items():
                if category not in selected:
                    continue
                for target in item.targets:
                    if (
                        category == "hf-cache"
                        and "raw" in selected
                        and Path(target.location).is_relative_to(
                            plan.directory / RAW_DIRNAME
                        )
                        and "raw" in removed
                    ):
                        continue
                    try:
                        if _remove_target(plan, category, target):
                            removed.add(category)
                            print(f"Removed {target.location}")
                        else:
                            print(f"Already absent: {target.location}")
                    except (OSError, RuntimeError, ValueError) as exc:
                        if category == "images":
                            failed_images.add(target.location)
                        print(
                            f"Could not remove {target.location}: {exc}",
                            file=sys.stderr,
                        )
                        status = 1
        finally:
            _record_cleanup(plan, selected, failed_images)
        return status


def _require_inspected(plan: CleanupPlan, category: str) -> None:
    item = plan.items[category]
    # If the unpinned shared cache cannot be identified, the scoped Hub cache is still safe.
    if item.error and not (category == "hf-cache" and item.targets):
        raise RuntimeError(f"Cannot clean {category}: {item.error}")


def run_cleanup(args: argparse.Namespace) -> int:
    try:
        if not args.dry_run and not args.yes:
            raise ValueError("Cleanup requires --yes")
        selected = set(args.delete)
        plan = make_cleanup_plan(args, selected or None)
        render_cleanup(plan, selected or None)
        return 0 if args.dry_run else apply_cleanup(plan, selected)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Cleanup failed: {exc}", file=sys.stderr)
        return 1
