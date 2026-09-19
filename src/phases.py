from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.constants import (
    BASE_DIR,
    EXTRACTED_DIRNAME,
    PARTS_DIRNAME,
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
    RAW_DIRNAME,
)
from src.docker import (
    confirm_unpushed,
    docker_image_exists,
    docker_image_size,
    ensure_dockerfiles,
    extract_image_parts,
    run_docker,
)
from src.files import (
    extracted_parts_exist,
    remove_file,
    restore_raw_file,
    restored_file_valid,
    split_raw_file,
)
from src.huggingface import download_model_file, list_model_files, parse_huggingface_url
from src.naming import docker_pull_tag, safe_path_name
from src.phase_state import (
    load_or_create_pull_state,
    parts_for_entry,
    parts_for_raw,
    reconcile_raw_files,
    replace_raw_parts,
)
from src.progress import format_elapsed as _format_elapsed
from src.progress import make_progress as _make_progress
from src.sizes import human_size
from src.state import (
    expected_raw_path,
    load_state,
    normalize_path,
    resolve_model_name,
    resolve_state_path,
    save_state,
)

PhaseHandler = Callable[[argparse.Namespace], None]


def _base_dir(args: argparse.Namespace) -> Path:
    custom = getattr(args, "model_weights_dir", None)
    if custom:
        return normalize_path(str(custom))
    return BASE_DIR


__all__ = [
    "PHASE_HANDLERS",
    "load_or_create_pull_state",
    "parts_for_entry",
    "parts_for_raw",
    "phase_pull_dockerhub",
    "phase_pull_llm",
    "phase_push_docker",
    "phase_restore_llm",
    "reconcile_raw_files",
    "replace_raw_parts",
]


def phase_pull_llm(args: argparse.Namespace) -> None:
    if not args.huggingface_link:
        raise ValueError(f"--huggingface-link is required for {PHASE_PULL_LLM}")

    phase_start = time.perf_counter()
    repo_id, revision = parse_huggingface_url(args.huggingface_link, args.revision)
    model_name = safe_path_name(args.model_name or repo_id.split("/", 1)[1])
    model_dir = _base_dir(args) / model_name
    raw_dir = model_dir / RAW_DIRNAME
    parts_dir = model_dir / PARTS_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)

    state = load_or_create_pull_state(
        model_dir,
        model_name,
        repo_id,
        revision,
        args.huggingface_link,
        args.max_part_size_bytes,
        args.max_docker_image_size_bytes,
    )
    files = list_model_files(repo_id, revision)
    raw_entries = reconcile_raw_files(state, files, raw_dir)
    save_state(model_dir, state)

    pending = [
        raw
        for raw in raw_entries
        if not (raw.get("split") and parts_for_raw(state, raw["filename"]))
    ]
    print(
        f"pull_llm [{model_name}]: {len(pending)}/{len(raw_entries)} files to process",
        flush=True,
    )
    with _make_progress("pull_llm") as progress:
        files_task = progress.add_task("files", total=max(len(raw_entries), 1))
        progress.update(files_task, completed=len(raw_entries) - len(pending))
        for raw in raw_entries:
            raw_rel = Path(raw["filename"])
            existing_parts = parts_for_raw(state, raw["filename"])
            if raw.get("split") and existing_parts:
                continue

            file_start = time.perf_counter()
            raw_path = resolve_state_path(model_dir, str(raw.get("path", "")))
            if not raw_path.exists() or not raw.get("downloaded"):
                print(f"Downloading {raw['filename']}", flush=True)
                raw_path = download_model_file(
                    repo_id, raw["filename"], revision, raw_dir
                )
                raw["path"] = expected_raw_path(raw["filename"])

            raw_size = raw_path.stat().st_size
            raw["size_bytes"] = raw_size
            raw["size"] = human_size(raw_size)
            raw["downloaded"] = True
            raw["deleted"] = False
            save_state(model_dir, state)

            split_task = progress.add_task(
                f"split {raw['filename']}", total=max(raw_size, 1)
            )

            def _on_split_bytes(n: int, _task: Any = split_task) -> None:
                progress.update(_task, advance=n)

            raw_sha, new_parts = split_raw_file(
                raw_path,
                raw_rel,
                parts_dir,
                args.max_part_size_bytes,
                progress_cb=_on_split_bytes,
            )
            progress.update(split_task, completed=raw_size)
            progress.remove_task(split_task)
            raw["sha256"] = raw_sha
            raw["split"] = True
            replace_raw_parts(state, raw["filename"], new_parts)
            save_state(model_dir, state)

            raw_deleted = False
            if not args.keep_raw:
                raw_deleted = remove_file(raw_path)
            raw["deleted"] = raw_deleted
            save_state(model_dir, state)
            progress.update(files_task, advance=1)
            print(
                f"pull_llm [{model_name}]: {raw['filename']} "
                f"done in {_format_elapsed(time.perf_counter() - file_start)}",
                flush=True,
            )
    print(
        f"pull_llm [{model_name}]: completed "
        f"{len(raw_entries)} files in {_format_elapsed(time.perf_counter() - phase_start)}",
        flush=True,
    )


def phase_push_docker(args: argparse.Namespace) -> None:
    phase_start = time.perf_counter()
    base_dir = _base_dir(args)
    model_name = resolve_model_name(args, PHASE_PUSH_DOCKER, base_dir)
    model_dir = base_dir / model_name
    state = load_state(model_dir)
    ensure_dockerfiles(
        model_dir, state, args.docker_namespace, args.max_docker_image_size_bytes
    )
    save_state(model_dir, state)

    part_by_name = {part["part_filename"]: part for part in state.get("parts", [])}
    entries = state.get("dockerfiles", [])
    print(
        f"push_docker [{model_name}]: {len(entries)} images to process",
        flush=True,
    )
    with _make_progress("push_docker") as progress:
        task = progress.add_task("images", total=max(len(entries), 1))
        for entry in entries:
            image_start = time.perf_counter()
            if (
                entry.get("built")
                and not entry.get("removed")
                and not docker_image_exists(entry["tag"])
            ):
                entry["built"] = False
                entry["removed"] = True
                save_state(model_dir, state)
            if not entry.get("built"):
                dockerfile_path = resolve_state_path(
                    model_dir, str(entry.get("dockerfile", ""))
                )
                run_docker(
                    [
                        "build",
                        "-f",
                        dockerfile_path.as_posix(),
                        "-t",
                        entry["tag"],
                        model_dir.as_posix(),
                    ]
                )
                entry["built"] = True
                entry["removed"] = False
                size_bytes = docker_image_size(entry["tag"])
                if size_bytes is not None:
                    entry["size_bytes"] = size_bytes
                    entry["size"] = human_size(size_bytes)
                save_state(model_dir, state)
            elif entry.get("size_bytes") is None and docker_image_exists(entry["tag"]):
                size_bytes = docker_image_size(entry["tag"])
                if size_bytes is not None:
                    entry["size_bytes"] = size_bytes
                    entry["size"] = human_size(size_bytes)
                    save_state(model_dir, state)
            if not entry.get("pushed"):
                run_docker(["push", entry["tag"]])
                entry["pushed"] = True
                save_state(model_dir, state)
            if not args.keep_images and not entry.get("removed"):
                if docker_image_exists(entry["tag"]):
                    run_docker(["rmi", entry["tag"]])
                entry["removed"] = True
                save_state(model_dir, state)
            if not args.keep_parts and entry.get("pushed"):
                for part_name in entry["included_parts"]:
                    part = part_by_name[part_name]
                    if not part.get("deleted"):
                        remove_file(
                            resolve_state_path(model_dir, str(part.get("path", "")))
                        )
                        part["deleted"] = True
                save_state(model_dir, state)
            progress.update(task, advance=1)
            print(
                f"push_docker [{model_name}]: {entry.get('label')} "
                f"done in {_format_elapsed(time.perf_counter() - image_start)}",
                flush=True,
            )
    print(
        f"push_docker [{model_name}]: completed "
        f"{len(entries)} images in {_format_elapsed(time.perf_counter() - phase_start)}",
        flush=True,
    )


def phase_pull_dockerhub(args: argparse.Namespace) -> None:
    phase_start = time.perf_counter()
    base_dir = _base_dir(args)
    model_name = resolve_model_name(args, PHASE_PULL_DOCKERHUB, base_dir)
    model_dir = base_dir / model_name
    state = load_state(model_dir)
    entries = state.get("dockerfiles", [])
    print(
        f"pull_dockerhub [{model_name}]: {len(entries)} images to process",
        flush=True,
    )
    decision: str | None = None
    with _make_progress("pull_dockerhub") as progress:
        task = progress.add_task("images", total=max(len(entries), 1))
        for entry in entries:
            image_start = time.perf_counter()
            if not entry.get("pushed"):
                should_skip, decision = confirm_unpushed(entry["tag"], decision)
                if should_skip:
                    progress.update(task, advance=1)
                    continue
                raise RuntimeError(f"Canceled because {entry['tag']} has pushed=false")
            pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
            if entry.get("pulled") and docker_image_exists(pull_tag):
                progress.update(task, advance=1)
                continue
            if not entry.get("pulled") or not docker_image_exists(pull_tag):
                pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
                run_docker(["pull", pull_tag])
                entry["pull_tag"] = pull_tag
                entry["pulled"] = True
                entry["removed"] = False
                save_state(model_dir, state)
            progress.update(task, advance=1)
            print(
                f"pull_dockerhub [{model_name}]: {entry.get('label')} "
                f"done in {_format_elapsed(time.perf_counter() - image_start)}",
                flush=True,
            )
    print(
        f"pull_dockerhub [{model_name}]: completed "
        f"{len(entries)} images in {_format_elapsed(time.perf_counter() - phase_start)}",
        flush=True,
    )


def phase_restore_llm(args: argparse.Namespace) -> None:
    phase_start = time.perf_counter()
    base_dir = _base_dir(args)
    model_name = resolve_model_name(args, PHASE_RESTORE_LLM, base_dir)
    model_dir = base_dir / model_name
    state = load_state(model_dir)
    part_by_name = {part["part_filename"]: part for part in state.get("parts", [])}

    entries = state.get("dockerfiles", [])
    print(
        f"restore_llm [{model_name}]: {len(entries)} images, "
        f"{len(state.get('raw', []))} files to process",
        flush=True,
    )
    with _make_progress("restore_llm") as progress:
        images_task = progress.add_task("images", total=max(len(entries), 1))
        for entry in entries:
            image_start = time.perf_counter()
            pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
            if not entry.get("pulled") or not docker_image_exists(pull_tag):
                run_docker(["pull", pull_tag])
                entry["pull_tag"] = pull_tag
                entry["pulled"] = True
                entry["removed"] = False
                save_state(model_dir, state)
            entry_parts = parts_for_entry(part_by_name, entry)
            if not entry.get("extracted") or not extracted_parts_exist(
                model_dir, entry_parts
            ):
                extract_image_parts(model_dir, entry, args.docker_pull_prefix)
                entry["extracted"] = True
                for part in entry_parts:
                    part["extracted"] = True
                save_state(model_dir, state)
            if not args.keep_images and not entry.get("removed"):
                if docker_image_exists(pull_tag):
                    run_docker(["rmi", pull_tag])
                entry["removed"] = True
                save_state(model_dir, state)
            progress.update(images_task, advance=1)
            print(
                f"restore_llm [{model_name}]: {entry.get('label')} "
                f"done in {_format_elapsed(time.perf_counter() - image_start)}",
                flush=True,
            )

        restore_by_raw = {item["raw_filename"]: item for item in state.get("restore", [])}
        parts_by_raw: dict[str, list[dict[str, Any]]] = {}
        for part in state.get("parts", []):
            parts_by_raw.setdefault(part["raw_filename"], []).append(part)

        raw_list = state.get("raw", [])
        files_task = progress.add_task("files", total=max(len(raw_list), 1))
        for raw in raw_list:
            if restore_by_raw.get(raw["filename"], {}).get(
                "restored"
            ) and restored_file_valid(model_dir, raw):
                progress.update(files_task, advance=1)
                continue
            file_start = time.perf_counter()
            total_bytes = int(raw.get("size_bytes", 0) or 0)
            restore_task = progress.add_task(
                f"restore {raw['filename']}", total=max(total_bytes, 1)
            )

            def _on_restore_bytes(n: int, _task: Any = restore_task) -> None:
                progress.update(_task, advance=n)

            restored = restore_raw_file(
                model_dir,
                raw,
                parts_by_raw.get(raw["filename"], []),
                progress_cb=_on_restore_bytes,
            )
            progress.update(restore_task, completed=max(total_bytes, 1))
            progress.remove_task(restore_task)
            state["restore"] = [
                item
                for item in state.get("restore", [])
                if item["raw_filename"] != raw["filename"]
            ]
            state["restore"].append(restored)
            for part in parts_by_raw.get(raw["filename"], []):
                part["restored"] = True
                if not args.keep_extracted_parts:
                    extracted = (
                        model_dir
                        / EXTRACTED_DIRNAME
                        / PARTS_DIRNAME
                        / part["part_filename"]
                    )
                    remove_file(extracted)
            save_state(model_dir, state)
            progress.update(files_task, advance=1)
            print(
                f"restore_llm [{model_name}]: {raw['filename']} "
                f"done in {_format_elapsed(time.perf_counter() - file_start)}",
                flush=True,
            )
    print(
        f"restore_llm [{model_name}]: completed "
        f"{len(entries)} images in {_format_elapsed(time.perf_counter() - phase_start)}",
        flush=True,
    )


PHASE_HANDLERS: dict[str, PhaseHandler] = {
    PHASE_PULL_LLM: phase_pull_llm,
    PHASE_PUSH_DOCKER: phase_push_docker,
    PHASE_PULL_DOCKERHUB: phase_pull_dockerhub,
    PHASE_RESTORE_LLM: phase_restore_llm,
}
