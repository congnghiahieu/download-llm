from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.constants import BASE_DIR, STATE_FILENAME
from src.naming import docker_repo_name, safe_path_name
from src.sizes import human_size


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_state(model_dir: Path) -> dict[str, Any]:
    state_path = model_dir / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with state_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_single_model_name() -> str | None:
    if not BASE_DIR.exists():
        return None
    candidates = [
        path.name for path in BASE_DIR.iterdir() if (path / STATE_FILENAME).exists()
    ]
    return candidates[0] if len(candidates) == 1 else None


def resolve_model_name(args: argparse.Namespace, phase: str) -> str:
    if args.model_name:
        return safe_path_name(args.model_name)
    inferred = infer_single_model_name()
    if inferred:
        return inferred
    raise ValueError(
        f"--model-name is required for {phase} unless exactly one model_weights/*/{STATE_FILENAME} exists"
    )


def save_state(model_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(model_dir / STATE_FILENAME, state)


def make_initial_state(
    model_name: str,
    repo_id: str,
    revision: str,
    url: str,
    max_part_size: int,
    max_docker_image_size: int,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "docker_repo_name": docker_repo_name(model_name),
        "repo_id": repo_id,
        "revision": revision,
        "url": url,
        "max_part_size_bytes": max_part_size,
        "max_part_size": human_size(max_part_size),
        "max_docker_image_size_bytes": max_docker_image_size,
        "max_docker_image_size": human_size(max_docker_image_size),
        "raw": [],
        "parts": [],
        "dockerfiles": [],
        "restore": [],
    }
