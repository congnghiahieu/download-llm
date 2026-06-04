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
    STATE_FILENAME,
)
from src.docker import (
    confirm_unpushed,
    docker_image_exists,
    ensure_dockerfiles,
    extract_image_parts,
    run_docker,
)
from src.files import (
    extracted_parts_exist,
    remove_file,
    restored_file_valid,
    restore_raw_file,
    sha256_file,
    split_raw_file,
)
from src.huggingface import download_model_file, list_model_files, parse_huggingface_url
from src.naming import docker_pull_tag, safe_path_name
from src.sizes import human_size
from src.state import load_state, make_initial_state, resolve_model_name, save_state

PhaseHandler = Callable[[argparse.Namespace], None]


def parts_for_raw(state: dict[str, Any], raw_filename: str) -> list[dict[str, Any]]:
    return [
        part for part in state.get("parts", []) if part.get("raw_filename") == raw_filename
    ]


def parts_for_entry(
    part_by_name: dict[str, dict[str, Any]], entry: dict[str, Any]
) -> list[dict[str, Any]]:
    return [part_by_name[name] for name in entry.get("included_parts", [])]


def replace_raw_parts(
    state: dict[str, Any], raw_filename: str, new_parts: list[dict[str, Any]]
) -> None:
    state["parts"] = [
        part for part in state.get("parts", []) if part.get("raw_filename") != raw_filename
    ]
    state["parts"].extend(new_parts)


def reconcile_raw_files(
    state: dict[str, Any], filenames: list[str], raw_dir: Path
) -> list[dict[str, Any]]:
    raw_by_name = {raw["filename"]: raw for raw in state.get("raw", [])}
    ordered = []
    for filename in filenames:
        raw_rel = Path(filename)
        raw = raw_by_name.get(raw_rel.as_posix())
        if raw is None:
            raw = {
                "filename": raw_rel.as_posix(),
                "path": (raw_dir / raw_rel).as_posix(),
                "downloaded": False,
                "split": False,
                "deleted": False,
            }
        else:
            raw.setdefault("path", (raw_dir / raw_rel).as_posix())
            raw.setdefault("downloaded", "sha256" in raw and "size_bytes" in raw)
            raw.setdefault("split", bool(parts_for_raw(state, raw["filename"])))
            raw.setdefault("deleted", False)
        ordered.append(raw)
    state["raw"] = ordered
    return ordered


def load_or_create_pull_state(
    model_dir: Path,
    model_name: str,
    repo_id: str,
    revision: str,
    url: str,
    max_part_size: int,
    max_docker_image_size: int,
) -> dict[str, Any]:
    if (model_dir / STATE_FILENAME).exists():
        state = load_state(model_dir)
        if (
            state.get("model_name") != model_name
            or state.get("repo_id") != repo_id
            or state.get("revision") != revision
        ):
            raise ValueError("Existing state.json does not match requested model")
        return state
    return make_initial_state(
        model_name, repo_id, revision, url, max_part_size, max_docker_image_size
    )


def phase_pull_llm(args: argparse.Namespace) -> None:
    if not args.huggingface_link:
        raise ValueError(f"--huggingface-link is required for {PHASE_PULL_LLM}")

    repo_id, revision = parse_huggingface_url(args.huggingface_link, args.revision)
    model_name = safe_path_name(args.model_name or repo_id.split("/", 1)[1])
    model_dir = BASE_DIR / model_name
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

    for raw in raw_entries:
        raw_rel = Path(raw["filename"])
        existing_parts = parts_for_raw(state, raw["filename"])
        if raw.get("split") and existing_parts:
            continue

        raw_path = Path(raw["path"])
        if not raw_path.exists() or not raw.get("downloaded"):
            print(f"Downloading {raw['filename']}", flush=True)
            raw_path = download_model_file(repo_id, raw["filename"], revision, raw_dir)
            raw["path"] = raw_path.as_posix()

        raw_size = raw_path.stat().st_size
        raw_sha = sha256_file(raw_path)
        raw["size_bytes"] = raw_size
        raw["size"] = human_size(raw_size)
        raw["sha256"] = raw_sha
        raw["downloaded"] = True
        raw["deleted"] = False
        save_state(model_dir, state)

        raw_sha, new_parts = split_raw_file(
            raw_path, raw_rel, parts_dir, args.max_part_size_bytes
        )
        raw["sha256"] = raw_sha
        raw["split"] = True
        replace_raw_parts(state, raw["filename"], new_parts)
        save_state(model_dir, state)

        raw_deleted = False
        if not args.keep_raw:
            raw_deleted = remove_file(raw_path)
        raw["deleted"] = raw_deleted
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
        if (
            entry.get("built")
            and not entry.get("removed")
            and not docker_image_exists(entry["tag"])
        ):
            entry["built"] = False
            entry["removed"] = True
            save_state(model_dir, state)
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
            entry["removed"] = False
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
                    remove_file(Path(part["path"]))
                    part["deleted"] = True
            save_state(model_dir, state)


def phase_pull_dockerhub(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, PHASE_PULL_DOCKERHUB)
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)
    decision: str | None = None
    for entry in state.get("dockerfiles", []):
        if not entry.get("pushed"):
            should_skip, decision = confirm_unpushed(entry["tag"], decision)
            if should_skip:
                continue
            raise RuntimeError(f"Canceled because {entry['tag']} has pushed=false")
        pull_tag = docker_pull_tag(entry, args.docker_pull_prefix)
        if entry.get("pulled") and docker_image_exists(pull_tag):
            continue
        if not entry.get("pulled") or not docker_image_exists(pull_tag):
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
    part_by_name = {part["part_filename"]: part for part in state.get("parts", [])}

    for entry in state.get("dockerfiles", []):
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

    restore_by_raw = {item["raw_filename"]: item for item in state.get("restore", [])}
    parts_by_raw: dict[str, list[dict[str, Any]]] = {}
    for part in state.get("parts", []):
        parts_by_raw.setdefault(part["raw_filename"], []).append(part)

    for raw in state.get("raw", []):
        if restore_by_raw.get(raw["filename"], {}).get(
            "restored"
        ) and restored_file_valid(model_dir, raw):
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
