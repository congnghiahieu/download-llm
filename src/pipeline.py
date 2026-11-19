from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import ExitStack, closing, nullcontext
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.constants import (
    EXTRACTED_DIRNAME,
    PARTS_DIRNAME,
    PHASE_DELIVER_LLM,
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
    RAW_DIRNAME,
    STATE_FILENAME,
)
from src.delivery import Delivery, previously_delivered, validate_delivery
from src.docker import (
    _expected_docker_entries,
    build_isolated_image,
    docker_image_exists,
    dockerfile_text,
    extract_image_parts,
    push_image_digest,
    remove_local_image,
    run_docker,
)
from src.files import (
    extracted_parts_exist,
    part_path_for_file,
    path_matches,
    remove_file,
    restore_raw_file,
    restored_file_valid,
    restored_record,
    split_raw_file,
)
from src.huggingface import (
    cached_download_bytes,
    download_snapshot_file,
    model_snapshot,
    parse_huggingface_url,
)
from src.naming import docker_pull_tag
from src.run_logging import log_file, logger
from src.scheduling import Scheduler
from src.sizes import human_size
from src.state import (
    expected_part_path,
    expected_raw_path,
    expected_restored_path,
    is_pull_push_complete,
    load_state,
    make_initial_state,
    model_lock,
    phase_progress,
    phase_status,
    resolve_state_path,
    save_state,
)

if TYPE_CHECKING:
    from src.planning import DeliveryDiskForecast

IMAGE_OVERHEAD = 32 * 1024**2  # base layers, filesystem and image metadata estimate
Record = dict[str, Any]


@dataclass
class Task:
    key: tuple[str, str]
    kind: str
    disk: dict[str, int]
    work: Callable[[], Any]
    commit: Callable[[Any], None]
    started: float = 0.0


def _validate_path(filename: str) -> Path:
    path = Path(filename)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != filename
        or ".." in path.parts
    ):
        raise ValueError(f"Invalid file path in model inventory: {filename!r}")
    return path


def _part_inventory(files: list[tuple[str, int]], max_part: int) -> list[Record]:
    inventory = []
    names: set[str] = set()
    for filename, size in files:
        path = _validate_path(filename)
        for index, offset in enumerate(range(0, size, max_part), 1):
            name = part_path_for_file(path, index).as_posix()
            if name in names:
                raise ValueError(f"Part filename collision: {name!r}")
            names.add(name)
            inventory.append(
                {
                    "raw_filename": filename,
                    "index": index,
                    "part_filename": name,
                    "path": expected_part_path(name),
                    "size_bytes": min(max_part, size - offset),
                }
            )
    return inventory


def _prepare_snapshot(
    args: argparse.Namespace, model_dir: Path, state: Record
) -> list[Record]:
    repo, revision = parse_huggingface_url(args.huggingface_link, args.revision)
    if state and (state.get("repo_id"), state.get("revision")) != (repo, revision):
        raise ValueError(
            f"{model_dir.name}: existing state does not match the requested model/revision"
        )
    snapshot = state.get("snapshot")
    if snapshot:
        commit, files = (
            snapshot["commit"],
            [(name, size) for name, size in snapshot["files"]],
        )
        if state["max_part_size_bytes"] != args.max_part_size_bytes:
            raise ValueError(
                f"{model_dir.name}: use the original --max-part-size or a new model directory"
            )
    else:
        if state.get("raw") or state.get("parts") or state.get("dockerfiles"):
            raise ValueError(
                f"{model_dir.name}: legacy download has no pinned revision; use a new model directory "
                "for pull_llm, or use existing state with push_docker/pull_dockerhub/restore_llm"
            )
        commit, files = model_snapshot(repo, revision)
        if not state:
            state.update(
                make_initial_state(
                    model_dir.name,
                    repo,
                    revision,
                    args.huggingface_link,
                    args.max_part_size_bytes,
                    args.max_docker_image_size_bytes,
                )
            )
        state["snapshot"] = {"commit": commit, "files": files}
    before = {raw["filename"]: raw for raw in state["raw"]}
    state["raw"] = [
        before.get(
            name,
            {
                "filename": name,
                "path": expected_raw_path(name),
                "size_bytes": size,
                "downloaded": False,
                "split": False,
                "deleted": False,
            },
        )
        for name, size in files
    ]
    return _part_inventory(files, args.max_part_size_bytes)


def _prepare_images(
    args: argparse.Namespace, state: Record, inventory: list[Record]
) -> None:
    was_complete = bool(state.get("publication", {}).get("complete"))
    old = {entry["tag"]: entry for entry in state["dockerfiles"]}
    expected = _expected_docker_entries(
        inventory,
        args.docker_namespace,
        state["docker_repo_name"],
        args.max_docker_image_size_bytes,
    )
    if not expected:
        raise ValueError(f"{state['model_name']}: no nonempty model parts to publish")
    snapshot = state.get("snapshot")
    if snapshot:
        recipe = f"v1:{state['repo_id']}:{snapshot['commit']}:{state['max_part_size_bytes']}:{args.max_docker_image_size_bytes}"
        version = hashlib.sha256(recipe.encode()).hexdigest()
        for entry in expected:
            entry["tag"] = f"{entry['tag']}-{version}"
        state["publication"] = {"complete": False, "commit": snapshot["commit"]}
    elif old and all(entry.get("pushed") for entry in old.values()):
        # Keep already-published legacy references usable, including legacy group order.
        return
    entries = []
    for entry in expected:
        previous = old.get(entry["tag"], {})
        if previous.get("included_parts") == entry["included_parts"]:
            entries.append({**previous, **entry})
        else:
            entries.append({**entry, "built": False, "pushed": False})
    state["dockerfiles"] = entries
    state["max_docker_image_size_bytes"] = args.max_docker_image_size_bytes
    state["max_docker_image_size"] = human_size(args.max_docker_image_size_bytes)
    if snapshot:
        state["publication"]["complete"] = was_complete and all(
            _pushed(entry, True) for entry in entries
        )


def _pushed(entry: Record, pinned: bool) -> bool:
    return bool(
        entry.get("pushed")
        and entry.get("built")
        and (not pinned or entry.get("digest"))
    )


def _validate_manifest(
    state: Record, inventory: list[Record], *, published: bool
) -> None:
    sizes: dict[str, int] = defaultdict(int)
    sources = {raw["filename"]: raw for raw in state["raw"]}
    included = [
        name for entry in state["dockerfiles"] for name in entry["included_parts"]
    ]
    part_names = [part["part_filename"] for part in inventory]
    if len(set(included)) != len(included) or sorted(included) != sorted(part_names):
        raise ValueError("Manifest does not map each part to exactly one Docker image")
    for part in inventory:
        if part["raw_filename"] not in sources:
            raise ValueError(
                f"Manifest contains a part without a source file: {part['part_filename']!r}"
            )
        if published and not re.fullmatch(r"[0-9a-f]{64}", part.get("sha256") or ""):
            raise ValueError(
                f"Missing or malformed checksum for part {part['part_filename']!r}"
            )
        sizes[part["raw_filename"]] += part["size_bytes"]
    for name, raw in sources.items():
        if sizes[name] != raw.get("size_bytes"):
            raise ValueError(
                f"Manifest parts do not match size of source file {name!r}"
            )
        if published and not re.fullmatch(r"[0-9a-f]{64}", raw.get("sha256") or ""):
            raise ValueError(f"Missing or malformed checksum for source file {name!r}")
    for entry in state["dockerfiles"]:
        if entry.get("digest") and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", entry["digest"]
        ):
            raise ValueError(f"Invalid Docker image digest for {entry['tag']!r}")


@dataclass
class ModelPipeline:
    args: argparse.Namespace
    directory: Path
    state: Record
    inventory: list[Record]
    cache_role: bool
    fingerprint: tuple[int, int, int] | None
    containerd_push: bool = False
    running: dict[tuple[str, str], tuple[Task, Future[Any]]] = field(
        default_factory=dict
    )
    send_done: bool = False
    receive_ready: bool = False
    failure: Exception | None = None
    restored: set[str] = field(default_factory=set)
    extracted: set[str] = field(default_factory=set)
    pulled: set[str] = field(default_factory=set)
    awaiting_extract: set[str] = field(default_factory=set)
    built: set[str] = field(default_factory=set)
    _verified: dict[Path, tuple[tuple[int, ...], str | None, bool]] = field(default_factory=dict)
    started: float = 0.0

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def downloading(self) -> bool:
        return PHASE_PULL_LLM in self.args.phases

    @property
    def publishing(self) -> bool:
        return PHASE_PUSH_DOCKER in self.args.phases

    @property
    def receiving(self) -> bool:
        return PHASE_PULL_DOCKERHUB in self.args.phases or self.restoring

    @property
    def restoring(self) -> bool:
        return PHASE_RESTORE_LLM in self.args.phases

    @property
    def delivering(self) -> bool:
        return PHASE_DELIVER_LLM in self.args.phases

    def save(self) -> None:
        save_state(self.directory, self.state)

    @cached_property
    def part_layout(self) -> dict[str, Record]:
        return {part["part_filename"]: part for part in self.inventory}

    def start(self) -> None:
        if state_fingerprint(self.directory) != self.fingerprint:
            raise RuntimeError(
                f"{self.name}: state changed after planning; create a new plan"
            )
        self.started = time.monotonic()
        logger.info("Pipeline started [model=%r]", self.name)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.publishing:
            for entry in self.state["dockerfiles"]:
                path = resolve_state_path(self.directory, entry["dockerfile"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    dockerfile_text(
                        [self.part_layout[name] for name in entry["included_parts"]]
                    ),
                    encoding="utf-8",
                )
                if (
                    not _pushed(entry, bool(self.state.get("snapshot")))
                    and entry.get("built")
                    and docker_image_exists(entry["tag"])
                ):
                    self.built.add(entry["tag"])
        if self.downloading or self.publishing:
            self.save()

    def raw_path(self, raw: Record) -> Path:
        return resolve_state_path(self.directory, raw["path"])

    def part_path(self, part: Record) -> Path:
        return resolve_state_path(self.directory, part["path"])

    def _available(self, path: Path, size: int, digest: str | None) -> bool:
        try:
            info = path.stat()
        except FileNotFoundError:
            return False
        if info.st_size != size:
            return False
        fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        cached = self._verified.get(path)
        if cached is not None and cached[:2] == (fingerprint, digest):
            return cached[2]
        ready = path_matches(path, size, digest)
        self._verified[path] = fingerprint, digest, ready
        return ready

    def raw_available(self, raw: Record) -> bool:
        return bool(raw.get("downloaded") or raw.get("sha256")) and self._available(
            self.raw_path(raw), raw["size_bytes"], raw.get("sha256")
        )

    def part_available(self, part: Record) -> bool:
        return self._available(self.part_path(part), part["size_bytes"], part.get("sha256"))

    def required_send_parts(self, built: set[str]) -> set[str]:
        if not self.publishing or self.args.keep_parts:
            return set(self.part_layout)
        return {
            name
            for entry in self.state["dockerfiles"]
            if not _pushed(entry, bool(self.state.get("snapshot"))) and entry["tag"] not in built
            for name in entry["included_parts"]
        }

    @cached_property
    def parts_by_raw(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for name, part in self.part_layout.items():
            result[part["raw_filename"]].add(name)
        return result

    def needs_split(self, raw: Record, ready: set[str], required: set[str]) -> bool:
        return not raw.get("split") or bool(self.missing_split_parts(raw, ready, required))

    def missing_split_parts(self, raw: Record, ready: set[str], required: set[str]) -> set[str]:
        names = self.parts_by_raw[raw["filename"]]
        if raw.get("split"):
            names = names & required
        return names - ready

    def output_path(self, raw: Record) -> Path:
        return self.directory / expected_restored_path(raw["filename"])

    def extracted_path(self, part: Record) -> Path:
        return (
            self.directory / EXTRACTED_DIRNAME / PARTS_DIRNAME / part["part_filename"]
        )

    def image_parts_size(self, entry: Record) -> int:
        return sum(
            self.part_layout[name]["size_bytes"] for name in entry["included_parts"]
        )

    def image_estimate(self, entry: Record) -> int:
        return max(
            entry.get("size_bytes") or 0, self.image_parts_size(entry) + IMAGE_OVERHEAD
        )

    def push_disk_demand(self, size: int) -> dict[str, int]:
        # The verified containerd store streams existing blobs to the registry.
        return {} if self.containerd_push else {"docker": size}

    def missing_download_bytes(self, raw: Record) -> int:
        size = raw["size_bytes"]
        if self.raw_available(raw):
            return 0
        commit = self.state["snapshot"]["commit"]
        return size - cached_download_bytes(
            self.state["repo_id"], commit, raw["filename"], size
        )

    def _download(self, raw: Record) -> None:
        name, size = raw["filename"], raw["size_bytes"]
        path = download_snapshot_file(
            self.state["repo_id"],
            name,
            self.state["snapshot"]["commit"],
            self.directory / RAW_DIRNAME,
        )
        if not path_matches(path, size, raw.get("sha256")):
            raise RuntimeError(f"Content mismatch for pinned Hugging Face file {name!r}")

    def _downloaded(self, raw: Record, _: Any) -> None:
        raw.update(downloaded=True, deleted=False, size=human_size(raw["size_bytes"]))
        self.save()

    def _split(self, raw: Record, missing: set[str]) -> tuple[str, list[Record]]:
        root = self.directory / PARTS_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".split-", dir=root) as tmp:
            staging = Path(tmp)
            sha, parts = split_raw_file(
                self.raw_path(raw), Path(raw["filename"]), staging, self.state["max_part_size_bytes"],
                write_part=missing.__contains__,
            )
            previous = {part["part_filename"]: part for part in self.state["parts"]}
            if raw.get("sha256") and raw["sha256"] != sha:
                raise RuntimeError(f"Regenerated file differs from manifest: {raw['filename']}")
            for part in parts:
                name = part["part_filename"]
                old_sha = previous.get(name, {}).get("sha256")
                expected = self.part_layout.get(name)
                if (expected is None or expected["size_bytes"] != part["size_bytes"]
                        or (old_sha and old_sha != part["sha256"])):
                    raise RuntimeError(f"Regenerated part differs from manifest: {name}")
            for part in parts:
                name = part["part_filename"]
                target = root / name
                if name in missing:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / name, target)
                part["deleted"] = not target.is_file()
        return sha, parts

    def _split_done(self, raw: Record, result: tuple[str, list[Record]]) -> None:
        sha, parts = result
        name = raw["filename"]
        actual = [(part["part_filename"], part["size_bytes"]) for part in parts]
        expected = [
            (part["part_filename"], part["size_bytes"])
            for part in self.inventory
            if part["raw_filename"] == name
        ]
        if actual != expected:
            raise RuntimeError(f"Split output differs from plan for {name!r}")
        raw.update(sha256=sha, downloaded=True, split=True, deleted=False)
        previous = {part["part_filename"]: part for part in self.state["parts"]}
        for part in parts:
            for field_name in ("extracted", "restored"):
                part[field_name] = previous.get(part["part_filename"], {}).get(field_name, False)
        self.state["parts"] = [
            part for part in self.state["parts"] if part["raw_filename"] != name
        ] + parts
        self.save()
        if not self.args.keep_raw:
            raw["deleted"] = (
                remove_file(self.raw_path(raw)) or not self.raw_path(raw).exists()
            )
            self.save()

    def _built(self, entry: Record, size: int | None) -> None:
        entry.update(built=True, removed=False)
        if size is not None:
            entry.update(size_bytes=size, size=human_size(size))
        self.built.add(entry["tag"])
        self.save()

    def _pushed_done(self, entry: Record, parts: list[Record], digest: str) -> None:
        entry.update(pushed=True, digest=digest)
        self.save()
        self._cleanup_pushed(entry, parts)

    def _cleanup_pushed(self, entry: Record, parts: list[Record]) -> None:
        changed = False
        if not self.args.keep_images and not entry.get("removed"):
            remove_local_image(entry["tag"])
            entry["removed"] = True
            changed = True
        if not self.args.keep_parts:
            for part in parts:
                if not part.get("deleted"):
                    remove_file(self.part_path(part))
                    part["deleted"] = True
                    changed = True
        if changed:
            self.save()

    def _send_tasks(self) -> Iterator[Task]:
        pinned = bool(self.state.get("snapshot"))
        parts = {part["part_filename"]: part for part in self.state["parts"]}
        if self.publishing:
            for entry in self.state["dockerfiles"]:
                tag = entry["tag"]
                ready = [
                    parts[name] for name in entry["included_parts"] if name in parts
                ]
                if _pushed(entry, pinned):
                    self._cleanup_pushed(entry, ready)
                    continue
                if tag in self.built:
                    yield Task(
                        ("push", tag),
                        "docker_push",
                        self.push_disk_demand(self.image_estimate(entry)),
                        partial(push_image_digest, tag),
                        partial(self._pushed_done, entry, ready),
                    )
                elif len(ready) == len(entry["included_parts"]) and all(self.part_available(part) for part in ready):
                    yield Task(
                        ("build", tag),
                        "docker_build",
                        {"docker": 2 * self.image_estimate(entry)},
                        partial(build_isolated_image, self.directory, entry, ready),
                        partial(self._built, entry),
                    )
        if self.downloading:
            required = self.required_send_parts(self.built)
            present = {name for name, part in parts.items() if name in required and self.part_available(part)}
            for raw in self.state["raw"]:
                split = self.needs_split(raw, present, required)
                if not split and not self.args.keep_raw:
                    continue
                name, size = raw["filename"], raw["size_bytes"]
                if self.raw_available(raw):
                    if split:
                        missing = self.missing_split_parts(raw, present, required)
                        yield Task(
                            ("split", name),
                            "split",
                            {"weights": sum(self.part_layout[part]["size_bytes"] for part in missing)},
                            partial(self._split, raw, missing),
                            partial(self._split_done, raw),
                        )
                    elif raw.get("deleted"):
                        raw["deleted"] = False
                        self.save()
                else:
                    # Cache may hard-link the materialized file, but reserve for copy fallback.
                    demand = {"weights": size}
                    if self.cache_role:
                        demand["hf_cache"] = self.missing_download_bytes(raw)
                    yield Task(
                        ("download", name),
                        "hf_download",
                        demand,
                        partial(self._download, raw),
                        partial(self._downloaded, raw),
                    )

    def _init_receive(self) -> None:
        if not is_pull_push_complete(self.state):
            raise RuntimeError(f"{self.name}: no complete pull/push manifest")
        _validate_manifest(self.state, self.state["parts"], published=True)
        parts = {part["part_filename"]: part for part in self.state["parts"]}
        if self.restoring:
            saved = {item["raw_filename"]: item for item in self.state["restore"]}
            changed = False
            for raw in self.state["raw"]:
                name = raw["filename"]
                record = saved.get(name)
                if previously_delivered(self, raw):
                    self.restored.add(name)
                elif restored_file_valid(self.directory, raw):
                    self.restored.add(name)
                    if record is None or not record.get("restored"):
                        saved[name] = restored_record(name, raw["size_bytes"], raw["sha256"])
                        changed = True
                    elif record.get("deleted"):
                        record["deleted"] = False
                        changed = True
                elif record and not record.get("deleted") and not self.output_path(raw).exists():
                    record["deleted"] = True
                    changed = True
            for part in parts.values():
                if part["raw_filename"] in self.restored:
                    continue
                available = extracted_parts_exist(self.directory, [part])
                if available:
                    self.extracted.add(part["part_filename"])
                if part.get("extracted") != available:
                    part["extracted"] = available
                    changed = True
            if changed:
                self.state["restore"] = list(saved.values())
                self.save()
        for entry in self.state["dockerfiles"]:
            reference = docker_pull_tag(entry, self.args.docker_pull_prefix)
            if (entry.get("pulled") or entry.get("digest")) and docker_image_exists(
                reference
            ):
                if not entry.get("pulled"):
                    self._pulled(entry, reference, None)
                else:
                    self.pulled.add(entry["tag"])
                    if self.restoring:
                        self.awaiting_extract.add(entry["tag"])
        self.receive_ready = True

    def _pulled(self, entry: Record, reference: str, _: Any) -> None:
        entry.update(pulled=True, pull_tag=reference, removed=False)
        self.pulled.add(entry["tag"])
        if self.restoring:
            self.awaiting_extract.add(entry["tag"])
        self.save()

    def _extract(self, entry: Record, parts: list[Record]) -> None:
        root = self.directory / EXTRACTED_DIRNAME / PARTS_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as tmp:
            staging = Path(tmp)
            extract_image_parts(
                self.directory, entry, self.args.docker_pull_prefix, target_dir=staging
            )
            for part in parts:
                source = staging / part["part_filename"]
                if not path_matches(source, part["size_bytes"], part["sha256"]):
                    raise RuntimeError(
                        f"Extracted parts do not match the manifest for {entry['tag']}"
                    )
            for part in parts:
                target = self.extracted_path(part)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / part["part_filename"], target)

    def _extracted(
        self, entry: Record, reference: str, parts: list[Record], _: Any
    ) -> None:
        for part in parts:
            part["extracted"] = True
            self.extracted.add(part["part_filename"])
        entry["extracted"] = True
        self.save()
        self._cleanup_extracted_image(entry, reference)

    def _cleanup_extracted_image(self, entry: Record, reference: str) -> None:
        self.awaiting_extract.discard(entry["tag"])
        if not self.args.keep_images and entry["tag"] in self.pulled:
            remove_local_image(reference)
            self.pulled.discard(entry["tag"])
            entry["removed"] = True
            self.save()

    def _restored(self, raw: Record, parts: list[Record], result: Record) -> None:
        name = raw["filename"]
        result["deleted"] = False
        self.state["restore"] = [
            item for item in self.state["restore"] if item["raw_filename"] != name
        ] + [result]
        self.restored.add(name)
        for part in parts:
            part["restored"] = True
        self.save()
        if not self.args.keep_extracted_parts:
            for part in parts:
                remove_file(self.extracted_path(part))
                self.extracted.discard(part["part_filename"])

    def _receive_tasks(self) -> Iterator[Task]:
        if not self.receive_ready:
            self._init_receive()
        parts = {part["part_filename"]: part for part in self.state["parts"]}
        raw_parts: dict[str, list[Record]] = defaultdict(list)
        for part in parts.values():
            raw_parts[part["raw_filename"]].append(part)
        if self.restoring:
            for raw in self.state["raw"]:
                name = raw["filename"]
                needed = raw_parts[name]
                if name not in self.restored and all(
                    part["part_filename"] in self.extracted for part in needed
                ):
                    yield Task(
                        ("restore", name),
                        "extract_restore",
                        {"weights": raw["size_bytes"]},
                        partial(
                            restore_raw_file,
                            self.directory,
                            raw,
                            [part.copy() for part in needed],
                        ),
                        partial(self._restored, raw, needed),
                    )
        for entry in self.state["dockerfiles"]:
            tag = entry["tag"]
            reference = docker_pull_tag(entry, self.args.docker_pull_prefix)
            needed = [parts[name] for name in entry["included_parts"]]
            if self.restoring:
                needed = [
                    part
                    for part in needed
                    if part["raw_filename"] not in self.restored
                    and part["part_filename"] not in self.extracted
                ]
                if not needed:
                    self._cleanup_extracted_image(entry, reference)
                    continue
                if tag in self.pulled:
                    yield Task(
                        ("extract", tag),
                        "extract_restore",
                        {"weights": self.image_parts_size(entry)},
                        partial(self._extract, entry, needed),
                        partial(self._extracted, entry, reference, needed),
                    )
                    continue
            elif tag in self.pulled:
                continue
            self.require_pull_digest(entry)
            yield Task(
                ("pull", tag),
                "docker_pull",
                {"docker": 2 * self.image_estimate(entry)},
                partial(run_docker, ["pull", reference]),
                partial(self._pulled, entry, reference),
            )

    def require_pull_digest(self, entry: Record) -> None:
        if self.state.get("cleanup_requires_pinned_pull") and not entry.get("digest"):
            raise ValueError(f"{self.name}: cannot reproduce {entry['tag']} after cleanup without a published digest")

    def tasks(self) -> list[Task]:
        if not self.send_done:
            ready = list(self._send_tasks())
            if ready:
                return ready
            if self.running:
                return []
            if self.downloading and not all(
                raw.get("split") for raw in self.state["raw"]
            ):
                raise RuntimeError(
                    f"{self.name}: download cannot complete with the current checkpoint"
                )
            if self.publishing and not all(
                _pushed(entry, bool(self.state.get("snapshot")))
                for entry in self.state["dockerfiles"]
            ):
                raise RuntimeError(
                    f"{self.name}: missing local parts required to build remaining images; "
                    "rerun pull_llm,push_docker with the original Hugging Face link, revision and part size"
                )
            if self.publishing and "publication" in self.state:
                if not phase_status(self.state)[PHASE_PULL_LLM]:
                    raise RuntimeError(
                        f"{self.name}: cannot publish before all source files are ready"
                    )
                _validate_manifest(self.state, self.state["parts"], published=True)
                if not self.state["publication"]["complete"]:
                    self.state["publication"]["complete"] = True
                    self.save()
            self.send_done = True
        return list(self._receive_tasks()) if self.receiving else []


def state_fingerprint(directory: Path) -> tuple[int, int, int] | None:
    try:
        info = (directory / STATE_FILENAME).stat()
        return info.st_ino, info.st_mtime_ns, info.st_size
    except FileNotFoundError:
        return None


def prepare_model(
    args: argparse.Namespace,
    directory: Path,
    cache_role: bool,
    *,
    containerd_push: bool = False,
) -> ModelPipeline:
    fingerprint = state_fingerprint(directory)
    state = load_state(directory) if fingerprint else {}
    downloading = PHASE_PULL_LLM in args.phases
    publishing = PHASE_PUSH_DOCKER in args.phases
    receiving = PHASE_PULL_DOCKERHUB in args.phases or PHASE_RESTORE_LLM in args.phases
    if downloading:
        inventory = _prepare_snapshot(args, directory, state)
    else:
        inventory = state["parts"]
    if publishing:
        if not downloading and not (
            state["raw"]
            and all(raw.get("split") and raw.get("downloaded") for raw in state["raw"])
        ):
            raise ValueError(
                f"{directory.name}: push_docker requires completed pull_llm state"
            )
        if state.get("snapshot") and not downloading:
            files = state["snapshot"]["files"]
            inventory = _part_inventory(files, state["max_part_size_bytes"])
        _prepare_images(args, state, inventory)
    delivering = PHASE_DELIVER_LLM in args.phases
    if (receiving or delivering) and not publishing and not is_pull_push_complete(state):
        raise ValueError(
            f"{directory.name}: the model needs a complete pull/push manifest"
        )
    if receiving or delivering:
        _validate_manifest(state, inventory, published=not publishing)
    for part in inventory:
        _validate_path(part["part_filename"])
    for raw in state["raw"]:
        _validate_path(raw["filename"])
    model = ModelPipeline(
        args,
        directory,
        state,
        inventory,
        cache_role,
        fingerprint,
        containerd_push=containerd_push,
    )
    if delivering:
        validate_delivery(model)
    return model


def run_models(
    models: list[ModelPipeline],
    max_workers: int,
    scheduler: Scheduler,
    forecast: DeliveryDiskForecast | None = None,
) -> None:
    waiting = iter(models)
    active: list[ModelPipeline] = []
    failures: list[tuple[str, Exception]] = []
    admission_error: Exception | None = None

    def stop_admission(exc: Exception) -> None:
        nonlocal admission_error
        admission_error = exc
        # Failed models skip discovery but still commit every completed future.
        for model in active:
            model.failure = model.failure or exc
        logger.error(
            "Stopping task admission; saving in-flight results before exit: %s", exc
        )

    # A single coordinator commits all state transitions, even for different models.
    delivery = Delivery(scheduler.mark_change) if any(model.delivering for model in models) else None
    delivery_context = closing(delivery) if delivery is not None else nullcontext()
    forecast_interval = 30 if forecast is not None else None
    with ExitStack() as locks, closing(scheduler), delivery_context:
        while True:
            generation = scheduler.generation
            if delivery is not None:
                delivery.collect()
            while (
                admission_error is None
                and len(active) < max_workers
                and (model := next(waiting, None)) is not None
            ):
                try:
                    model.directory.mkdir(parents=True, exist_ok=True)
                    locks.enter_context(model_lock(model.directory))
                    model.start()
                except Exception as exc:  # noqa: BLE001 - preserve failures per model
                    failures.append((model.name, exc))
                    log_file(
                        logging.ERROR,
                        "Pipeline start failed [model=%r, storage=%r]",
                        model.name,
                        scheduler.disk_snapshot(),
                        exc_info=True,
                    )
                else:
                    active.append(model)
                    if delivery is not None and model.delivering:
                        delivery.add(model)
            if not active:
                if delivery is not None:
                    delivery.submit_next()
                    if delivery.running is not None:
                        if forecast is not None:
                            forecast.report(delivery, scheduler)
                        scheduler.wait_for_change(generation, external=True, timeout=forecast_interval)
                        continue
                break
            progressed = False
            candidates: list[tuple[int, ModelPipeline, Task]] = []
            for model in active[:]:
                for key, (task, future) in list(model.running.items()):
                    if future.done():
                        model.running.pop(key)
                        try:
                            task.commit(future.result())
                            if forecast is not None and key[0] == "restore":
                                forecast.restored(model, key[1])
                            logger.info(
                                "Task %s finished [model=%r, item=%r] (%.2fs)",
                                task.kind,
                                model.name,
                                key[1],
                                time.monotonic() - task.started,
                            )
                        except Exception as exc:  # noqa: BLE001 - preserve failures per model
                            model.failure = model.failure or exc
                            logger.error(
                                "Task %s failed [model=%r, item=%r] (%s)",
                                task.kind,
                                model.name,
                                key[1],
                                type(exc).__name__,
                            )
                            log_file(
                                logging.ERROR,
                                "Task failure details [model=%r, task=%s, item=%r, elapsed=%.2fs, demand=%r, storage=%r, stderr_tail=%r]",
                                model.name,
                                task.kind,
                                key[1],
                                time.monotonic() - task.started,
                                task.disk,
                                scheduler.disk_snapshot(),
                                getattr(exc, "stderr", None),
                                exc_info=True,
                            )
                        progressed = True
                if model.failure is None:
                    try:
                        ready = model.tasks()
                    except Exception as exc:  # noqa: BLE001 - preserve failures per model
                        model.failure = exc
                        ready = []
                        log_file(
                            logging.ERROR,
                            "Pipeline scheduling failed [model=%r, storage=%r]",
                            model.name,
                            scheduler.disk_snapshot(),
                            exc_info=True,
                        )
                    for task in ready:
                        if task.key not in model.running:
                            priority = {
                                "restore": 0,
                                "extract": 1,
                                "push": 1,
                                "build": 2,
                                "split": 2,
                            }.get(task.key[0], 3)
                            candidates.append((priority, model, task))
                else:
                    ready = []
                if not model.running and not ready:
                    if model.failure:
                        failures.append((model.name, model.failure))
                        logger.error(
                            "Pipeline failed [model=%r] (%s)",
                            model.name,
                            type(model.failure).__name__,
                        )
                        log_file(
                            logging.INFO,
                            "Checkpoint at failure [model=%r, progress=%r]",
                            model.name,
                            phase_progress(model.state),
                        )
                    else:
                        logger.info(
                            "Pipeline finished [model=%r] (%.2fs)",
                            model.name,
                            time.monotonic() - model.started,
                        )
                    active.remove(model)
                    progressed = True
            # Do not let pulls run indefinitely ahead of extract. Existing local images
            # may exceed this window, but no new pull is admitted until they drain.
            buffered = sum(
                len(model.awaiting_extract)
                + sum(key[0] == "pull" for key in model.running)
                for model in active
                if model.restoring
            )
            window = (
                scheduler.limits["docker_pull"] + scheduler.limits["extract_restore"]
            )
            if delivery is not None:
                scheduler.last_disk_block = None
            for _, model, task in sorted(candidates, key=lambda value: value[0]):
                buffered_pull = task.kind == "docker_pull" and model.restoring
                if buffered_pull and buffered >= window:
                    continue
                task.started = time.monotonic()
                try:
                    future = scheduler.submit(task.kind, task.disk, task.work)
                except Exception as exc:  # noqa: BLE001 - drain before surfacing admission failure
                    stop_admission(exc)
                    progressed = True
                    break
                if future is not None:
                    buffered += int(buffered_pull)
                    model.running[task.key] = task, future
                    log_file(
                        logging.INFO,
                        "Task %s admitted [model=%r, item=%r, demand=%r]",
                        task.kind,
                        model.name,
                        task.key[1],
                        task.disk,
                    )
                    progressed = True
                    if not scheduler.overlap:
                        break
            if delivery is not None:
                delivery.submit_next()
                if forecast is not None:
                    forecast.report(delivery, scheduler)
            if not progressed:
                if admission_error is not None:
                    wait(
                        [
                            future
                            for model in active
                            for _, future in model.running.values()
                        ],
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    try:
                        scheduler.wait_for_change(generation, timeout=forecast_interval)
                    except Exception as exc:  # noqa: BLE001 - preserve checkpoints on scheduler failure
                        if delivery is not None and forecast is not None:
                            forecast.report(delivery, scheduler, blocked=True)
                        stop_admission(exc)
        if delivery is not None:
            if forecast is not None:
                forecast.report(delivery, scheduler, force=True)
            delivery.finish()
            failures.extend(delivery.failures.items())
    if failures:
        from src.execution import build_retry_command

        failed = {name for name, _ in failures}
        print("Retry each failed model individually:", flush=True)
        for model in models:
            if model.name in failed:
                retry = model.args
                if delivery is not None and model.name in delivery.failures and model.failure is None:
                    retry = argparse.Namespace(**(vars(retry) | {"phases": [PHASE_DELIVER_LLM]}))
                command = build_retry_command(retry)
                print(f"  {command}", flush=True)
                log_file(
                    logging.INFO, "Retry command [model=%r]: %s", model.name, command
                )
        details = "; ".join(f"{name}: {exc}" for name, exc in failures)
        raise RuntimeError(
            f"{len(failed)} model(s) failed; rerun to resume: {details}"
        ) from failures[0][1]
