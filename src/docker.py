from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.constants import (
    ALPINE_IMAGE_NAME,
    CONFIRM_NO,
    CONFIRM_NO_ALL,
    CONFIRM_NO_ALL_INPUT,
    CONFIRM_YES,
    CONFIRM_YES_ALL,
    CONFIRM_YES_ALL_INPUT,
    DOCKERFILES_DIRNAME,
    EXTRACTED_DIRNAME,
    MODEL_PARTS_CONTAINER_DIR,
    PARTS_DIRNAME,
)
from src.naming import docker_pull_tag, docker_repo_name, part_label


def run_docker(args: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(["docker", *args]), flush=True)
    return subprocess.run(["docker", *args], check=True, text=True)


def docker_image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0


def group_parts(
    parts: list[dict[str, Any]], max_image_size: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for part in parts:
        size = int(part["size_bytes"])
        if size > max_image_size:
            raise ValueError(
                f"Part {part['part_filename']} is larger than max docker image size"
            )
        if current and current_size + size > max_image_size:
            groups.append(current)
            current = []
            current_size = 0
        current.append(part)
        current_size += size
    if current:
        groups.append(current)
    return groups


def dockerfile_text(group: list[dict[str, Any]]) -> str:
    lines = [f"FROM {ALPINE_IMAGE_NAME}"]
    for part in group:
        part_path = part["part_filename"]
        lines.append(
            f"COPY {PARTS_DIRNAME}/{part_path} {MODEL_PARTS_CONTAINER_DIR}/{part_path}"
        )
    return "\n".join(lines) + "\n"


def ensure_dockerfiles(
    model_dir: Path,
    state: dict[str, Any],
    namespace: str,
    max_image_size: int,
) -> None:
    if state.get("dockerfiles"):
        return
    groups = group_parts(state.get("parts", []), max_image_size)
    dockerfiles_dir = model_dir / DOCKERFILES_DIRNAME
    entries = []
    repo = state.get("docker_repo_name") or docker_repo_name(state["model_name"])

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{DOCKERFILES_DIRNAME}.", dir=model_dir))
    try:
        for index, group in enumerate(groups, start=1):
            label = part_label(index)
            dockerfile = tmp_dir / f"Dockerfile.{label}"
            dockerfile.write_text(dockerfile_text(group), encoding="utf-8")
            entries.append(
                {
                    "index": index,
                    "label": label,
                    "dockerfile": (dockerfiles_dir / f"Dockerfile.{label}").as_posix(),
                    "tag": f"{namespace}/{repo}:{label}",
                    "included_parts": [part["part_filename"] for part in group],
                    "built": False,
                    "pushed": False,
                    "pulled": False,
                    "removed": False,
                    "extracted": False,
                }
            )
        if dockerfiles_dir.exists():
            shutil.rmtree(dockerfiles_dir)
        os.replace(tmp_dir, dockerfiles_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    state["dockerfiles"] = entries


def confirm_unpushed(tag: str, decision: str | None) -> tuple[bool, str | None]:
    if decision == CONFIRM_YES_ALL:
        return True, decision
    if decision == CONFIRM_NO_ALL:
        return False, decision
    while True:
        print(f"ERROR: {tag} has pushed=false in state.json", file=sys.stderr)
        answer = input("Skip this unpushed image and continue? [y/n/ya/no]: ").strip().lower()
        if answer == CONFIRM_YES:
            return True, decision
        if answer == CONFIRM_NO:
            return False, decision
        if answer == CONFIRM_YES_ALL_INPUT:
            return True, CONFIRM_YES_ALL
        if answer == CONFIRM_NO_ALL_INPUT:
            return False, CONFIRM_NO_ALL
        print("Use y, n, ya, or no.", file=sys.stderr)


def extract_image_parts(
    model_dir: Path, entry: dict[str, Any], docker_pull_prefix: str | None = None
) -> None:
    container_name = f"download-llm-{entry['label']}-{os.getpid()}"
    image_tag = docker_pull_tag(entry, docker_pull_prefix)
    created = False
    try:
        run_docker(["create", "--name", container_name, image_tag])
        created = True
        target_dir = model_dir / EXTRACTED_DIRNAME / PARTS_DIRNAME
        target_dir.mkdir(parents=True, exist_ok=True)
        run_docker(
            [
                "cp",
                f"{container_name}:{MODEL_PARTS_CONTAINER_DIR}/.",
                target_dir.as_posix(),
            ]
        )
    finally:
        if created:
            subprocess.run(["docker", "rm", container_name], check=False, text=True)
