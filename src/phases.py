from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

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
    ensure_dockerfiles,
    extract_image_parts,
    run_docker,
)
from src.files import remove_file, restore_raw_file, split_raw_file
from src.huggingface import download_model_file, list_model_files, parse_huggingface_url
from src.naming import docker_pull_tag, safe_path_name
from src.sizes import human_size
from src.state import load_state, make_initial_state, resolve_model_name, save_state

PhaseHandler = Callable[[argparse.Namespace], None]


def phase_pull_llm(args: argparse.Namespace) -> None:
    if not args.huggingface_link:
        raise ValueError(f"--huggingface-link is required for {PHASE_PULL_LLM}")

    repo_id, revision = parse_huggingface_url(args.huggingface_link, args.revision)
    model_name = safe_path_name(args.model_name or repo_id.split("/", 1)[1])
    model_dir = BASE_DIR / model_name
    raw_dir = model_dir / RAW_DIRNAME
    parts_dir = model_dir / PARTS_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)

    state = make_initial_state(
        model_name,
        repo_id,
        revision,
        args.huggingface_link,
        args.max_part_size_bytes,
        args.max_docker_image_size_bytes,
    )
    save_state(model_dir, state)

    files = list_model_files(repo_id, revision)
    for filename in files:
        print(f"Downloading {filename}", flush=True)
        raw_path = download_model_file(repo_id, filename, revision, raw_dir)
        raw_rel = Path(filename)

        raw_size = raw_path.stat().st_size
        raw_sha, new_parts = split_raw_file(
            raw_path, raw_rel, parts_dir, args.max_part_size_bytes
        )
        raw_deleted = False
        if not args.keep_raw:
            raw_deleted = remove_file(raw_path)

        state["raw"].append(
            {
                "filename": raw_rel.as_posix(),
                "path": raw_path.as_posix(),
                "size_bytes": raw_size,
                "size": human_size(raw_size),
                "sha256": raw_sha,
                "deleted": raw_deleted,
            }
        )
        state["parts"].extend(new_parts)
        save_state(model_dir, state)


def phase_push_docker(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, PHASE_PUSH_DOCKER)
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)
    ensure_dockerfiles(
        model_dir, state, args.docker_namespace, args.max_docker_image_size_bytes
    )
    save_state(model_dir, state)

    part_by_name = {part["part_filename"]: part for part in state.get("parts", [])}
    for entry in state["dockerfiles"]:
        if not entry.get("built"):
            run_docker(
                [
                    "build",
                    "-f",
                    entry["dockerfile"],
                    "-t",
                    entry["tag"],
                    model_dir.as_posix(),
                ]
            )
            entry["built"] = True
            save_state(model_dir, state)
        if not entry.get("pushed"):
            run_docker(["push", entry["tag"]])
            entry["pushed"] = True
            save_state(model_dir, state)
        if not args.keep_images and not entry.get("removed"):
            run_docker(["rmi", entry["tag"]])
            entry["removed"] = True
            save_state(model_dir, state)
        if not args.keep_parts:
            for part_name in entry["included_parts"]:
                part = part_by_name[part_name]
                if not part.get("deleted"):
                    part["deleted"] = remove_file(Path(part["path"]))
            save_state(model_dir, state)


def phase_pull_dockerhub(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, PHASE_PULL_DOCKERHUB)
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)
    decision: str | None = None
    for entry in state.get("dockerfiles", []):
        if not entry.get("pushed"):
            should_pull, decision = confirm_unpushed(entry["tag"], decision)
            if not should_pull:
                continue
        if not entry.get("pulled"):
            pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
            run_docker(["pull", pull_tag])
            entry["pull_tag"] = pull_tag
            entry["pulled"] = True
            entry["removed"] = False
            save_state(model_dir, state)


def phase_restore_llm(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, PHASE_RESTORE_LLM)
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)

    for entry in state.get("dockerfiles", []):
        if not entry.get("pulled"):
            pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
            run_docker(["pull", pull_tag])
            entry["pull_tag"] = pull_tag
            entry["pulled"] = True
            save_state(model_dir, state)
        if not entry.get("extracted"):
            extract_image_parts(model_dir, entry, args.docker_pull_prefix)
            entry["extracted"] = True
            save_state(model_dir, state)
        if not args.keep_images and not entry.get("removed"):
            pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
            run_docker(["rmi", pull_tag])
            entry["removed"] = True
            save_state(model_dir, state)

    restore_by_raw = {item["raw_filename"]: item for item in state.get("restore", [])}
    parts_by_raw: dict[str, list[dict[str, Any]]] = {}
    for part in state.get("parts", []):
        parts_by_raw.setdefault(part["raw_filename"], []).append(part)

    for raw in state.get("raw", []):
        if restore_by_raw.get(raw["filename"], {}).get("restored"):
            continue
        restored = restore_raw_file(
            model_dir, raw, parts_by_raw.get(raw["filename"], [])
        )
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


PHASE_HANDLERS: dict[str, PhaseHandler] = {
    PHASE_PULL_LLM: phase_pull_llm,
    PHASE_PUSH_DOCKER: phase_push_docker,
    PHASE_PULL_DOCKERHUB: phase_pull_dockerhub,
    PHASE_RESTORE_LLM: phase_restore_llm,
}
