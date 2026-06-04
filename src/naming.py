from __future__ import annotations

import re
from typing import Any

DEFAULT_NAME = "model"
PATH_SAFE_PATTERN = r"[^A-Za-z0-9_.-]+"
DOCKER_REPO_SAFE_PATTERN = r"[^a-z0-9_.-]+"


def join_docker_prefix(prefix: str | None, tag: str) -> str:
    if not prefix:
        return tag
    normalized_prefix = prefix.rstrip("/")
    if tag == normalized_prefix or tag.startswith(f"{normalized_prefix}/"):
        return tag
    return f"{normalized_prefix}/{tag}"


def docker_pull_tag(entry: dict[str, Any], prefix: str | None = None) -> str:
    tag = entry.get("pull_tag") or entry["tag"]
    return join_docker_prefix(prefix, tag)


def safe_path_name(value: str) -> str:
    safe = re.sub(PATH_SAFE_PATTERN, "-", value).strip("-._")
    return safe or DEFAULT_NAME


def docker_repo_name(value: str) -> str:
    safe = re.sub(DOCKER_REPO_SAFE_PATTERN, "-", value.lower()).strip("-._")
    return safe or DEFAULT_NAME


def part_label(index: int) -> str:
    return f"part{index:04d}"
