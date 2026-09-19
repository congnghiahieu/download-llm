from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.constants import BASE_DIR, STATE_FILENAME
from src.files import remove_file
from src.state import (
    expected_part_path,
    expected_raw_path,
    load_state,
    make_initial_state,
    normalize_path,
    resolve_state_path,
)


def base_dir(args: argparse.Namespace) -> Path:
    custom = getattr(args, "model_weights_dir", None)
    if custom:
        return normalize_path(str(custom))
    return BASE_DIR


def parts_for_raw(state: dict[str, Any], raw_filename: str) -> list[dict[str, Any]]:
    return [
        part
        for part in state.get("parts", [])
        if part.get("raw_filename") == raw_filename
    ]


def parts_for_entry(
    part_by_name: dict[str, dict[str, Any]], entry: dict[str, Any]
) -> list[dict[str, Any]]:
    return [part_by_name[name] for name in entry.get("included_parts", [])]


def replace_raw_parts(
    state: dict[str, Any], raw_filename: str, new_parts: list[dict[str, Any]]
) -> None:
    state["parts"] = [
        part
        for part in state.get("parts", [])
        if part.get("raw_filename") != raw_filename
    ]
    state["parts"].extend(new_parts)


def reconcile_raw_files(
    state: dict[str, Any], filenames: list[str], raw_dir: Path
) -> list[dict[str, Any]]:
    model_dir = raw_dir.parent
    raw_by_name = {raw["filename"]: raw for raw in state.get("raw", [])}
    ordered = []
    for filename in filenames:
        raw_rel = Path(filename)
        raw = raw_by_name.get(raw_rel.as_posix())
        expected_path = expected_raw_path(raw_rel.as_posix())
        if raw is None:
            raw = {
                "filename": raw_rel.as_posix(),
                "path": expected_path,
                "downloaded": False,
                "split": False,
                "deleted": False,
            }
        else:
            raw.setdefault("path", expected_path)
            raw.setdefault("downloaded", "sha256" in raw and "size_bytes" in raw)
            raw.setdefault("split", bool(parts_for_raw(state, raw["filename"])))
            raw.setdefault("deleted", False)
            if not raw.get("path"):
                raw["path"] = expected_path
        ordered.append(raw)
    state["raw"] = ordered
    valid_names = {raw["filename"] for raw in ordered}
    orphan_parts = [
        part
        for part in state.get("parts", [])
        if part.get("raw_filename") not in valid_names
    ]
    if orphan_parts:
        for part in orphan_parts:
            stored = str(part.get("path", "") or "")
            if stored:
                candidate = resolve_state_path(
                    model_dir, stored or expected_part_path(str(part.get("part_filename", "")))
                )
                remove_file(candidate)
            else:
                fallback = model_dir / expected_part_path(
                    str(part.get("part_filename", ""))
                )
                remove_file(fallback)
        state["parts"] = [
            part
            for part in state.get("parts", [])
            if part.get("raw_filename") in valid_names
        ]
        state["restore"] = [
            item
            for item in state.get("restore", [])
            if item.get("raw_filename") in valid_names
        ]
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
