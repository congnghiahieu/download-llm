from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.constants import (
    BASE_DIR,
    DOCKERFILES_DIRNAME,
    EXTRACTED_DIRNAME,
    PARTS_DIRNAME,
    RAW_DIRNAME,
    RESTORED_DIRNAME,
    STATE_FILENAME,
)
from src.naming import docker_repo_name
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
        state = normalize_state(json.load(file))
    return migrate_state_paths(model_dir, state)


def resolve_state_path(model_dir: Path, stored: str) -> Path:
    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate
    resolved = model_dir / candidate
    if resolved.exists():
        return resolved
    if candidate.exists():
        return candidate
    return resolved


def expected_raw_path(filename: str) -> str:
    return f"{RAW_DIRNAME}/{Path(filename).as_posix()}"


def expected_part_path(part_filename: str) -> str:
    return f"{PARTS_DIRNAME}/{Path(part_filename).as_posix()}"


def expected_dockerfile_path(label: str) -> str:
    return f"{DOCKERFILES_DIRNAME}/Dockerfile.{label}"


def expected_restored_path(raw_filename: str) -> str:
    return f"{EXTRACTED_DIRNAME}/{RESTORED_DIRNAME}/{Path(raw_filename).as_posix()}"


def migrate_state_paths(model_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    for raw in state.get("raw", []):
        filename = str(raw.get("filename", ""))
        if not filename:
            continue
        stored = str(raw.get("path", "") or "")
        expected = expected_raw_path(filename)
        if not stored:
            raw["path"] = expected
            continue
        candidate = Path(stored)
        if candidate.is_absolute():
            try:
                raw["path"] = candidate.relative_to(model_dir).as_posix()
            except ValueError:
                raw["path"] = expected
        elif candidate.as_posix() != expected and (
            candidate.as_posix().endswith(expected)
            or not (model_dir / candidate).exists()
        ):
            raw["path"] = expected
    for part in state.get("parts", []):
        part_filename = str(part.get("part_filename", ""))
        if not part_filename:
            continue
        stored = str(part.get("path", "") or "")
        expected = expected_part_path(part_filename)
        if not stored:
            part["path"] = expected
            continue
        candidate = Path(stored)
        if candidate.is_absolute():
            try:
                part["path"] = candidate.relative_to(model_dir).as_posix()
            except ValueError:
                part["path"] = expected
        elif candidate.as_posix() != expected and not (model_dir / candidate).exists():
            part["path"] = expected
    for entry in state.get("dockerfiles", []):
        stored = str(entry.get("dockerfile", "") or "")
        label = str(entry.get("label", "") or "")
        if not stored or not label:
            continue
        expected = expected_dockerfile_path(label)
        candidate = Path(stored)
        if candidate.is_absolute():
            try:
                entry["dockerfile"] = candidate.relative_to(model_dir).as_posix()
            except ValueError:
                entry["dockerfile"] = expected
        elif candidate.as_posix() != expected and not (model_dir / candidate).exists():
            entry["dockerfile"] = expected
    for item in state.get("restore", []):
        raw_filename = str(item.get("raw_filename", ""))
        if not raw_filename:
            continue
        stored = str(item.get("restored_path", "") or "")
        expected = expected_restored_path(raw_filename)
        if not stored:
            item["restored_path"] = expected
            continue
        candidate = Path(stored)
        if candidate.is_absolute():
            try:
                item["restored_path"] = candidate.relative_to(model_dir).as_posix()
            except ValueError:
                item["restored_path"] = expected
        elif candidate.as_posix() != expected and not (model_dir / candidate).exists():
            item["restored_path"] = expected
    return state


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault("raw", [])
    state.setdefault("parts", [])
    state.setdefault("dockerfiles", [])
    state.setdefault("restore", [])

    parts_by_raw: dict[str, list[dict[str, Any]]] = {}
    for part in state["parts"]:
        part.setdefault("deleted", False)
        part.setdefault("extracted", False)
        part.setdefault("restored", False)
        parts_by_raw.setdefault(part.get("raw_filename", ""), []).append(part)

    for raw in state["raw"]:
        raw.setdefault("downloaded", "sha256" in raw and "size_bytes" in raw)
        raw.setdefault("split", bool(parts_by_raw.get(raw.get("filename", ""))))
        raw.setdefault("deleted", False)

    for entry in state["dockerfiles"]:
        entry.setdefault("built", False)
        entry.setdefault("pushed", False)
        entry.setdefault("pulled", False)
        entry.setdefault("removed", False)
        entry.setdefault("extracted", False)

    for item in state["restore"]:
        item.setdefault("restored", False)
    return state


def normalize_path(path: Path | str) -> Path:
    try:
        return Path(str(path)).expanduser().resolve()
    except OSError:
        return Path(str(path)).expanduser()


def effective_base_dir(args: argparse.Namespace | None = None) -> Path:
    if args is not None:
        custom = getattr(args, "model_weights_dir", None)
        if custom:
            return normalize_path(str(custom))
    return BASE_DIR


def infer_single_model_name(base_dir: Path | None = None) -> str | None:
    root = base_dir or BASE_DIR
    if not root.exists():
        return None
    candidates = [
        path.name for path in root.iterdir() if (path / STATE_FILENAME).exists()
    ]
    return candidates[0] if len(candidates) == 1 else None


def resolve_model_name(
    args: argparse.Namespace, phase: str, base_dir: Path | None = None
) -> str:
    names = resolve_model_names(args, phase, base_dir=base_dir)
    return names[0]


def split_multi_values(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                items.append(stripped)
    return items


def get_model_names(args: argparse.Namespace) -> list[str]:
    if getattr(args, "model_names", None) is not None:
        return split_multi_values(args.model_names)
    return split_multi_values(
        [args.model_name] if getattr(args, "model_name", None) else []
    )


def get_huggingface_links(args: argparse.Namespace) -> list[str]:
    if getattr(args, "huggingface_links", None) is not None:
        return split_multi_values(args.huggingface_links)
    return split_multi_values(
        [args.huggingface_link] if getattr(args, "huggingface_link", None) else []
    )


def list_model_dirs(base_dir: Path | None = None) -> list[Path]:
    root = base_dir or BASE_DIR
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if (path / STATE_FILENAME).exists()])


def resolve_model_alias(alias: str, base_dir: Path | None = None) -> str:
    root = base_dir or BASE_DIR
    wanted = alias.strip()
    if not wanted:
        raise ValueError("Empty --model-name value")
    wanted_lower = wanted.lower()
    matches: list[str] = []
    for model_dir in list_model_dirs(root):
        try:
            state = load_state(model_dir)
        except Exception:  # noqa: BLE001, S112 - invalid state must not fail alias scan
            continue
        candidates_ci = [
            str(state.get("model_name", "")).strip().lower(),
            str(state.get("docker_repo_name", "")).strip().lower(),
            model_dir.name.strip().lower(),
        ]
        candidates_exact = [
            str(state.get("repo_id", "")).strip(),
            str(state.get("url", "")).strip().rstrip("/"),
        ]
        if wanted_lower in [c for c in candidates_ci if c] or wanted.strip().rstrip(
            "/"
        ) in [c for c in candidates_exact if c]:
            matches.append(model_dir.name)
    if not matches:
        available = [path.name for path in list_model_dirs(root)]
        raise ValueError(
            f"Invalid --model-name '{alias}'. No model_weights/*/{STATE_FILENAME} "
            f"matches model_name, docker_repo_name, repo_id, or url. "
            f"Available: {', '.join(available) if available else '(none)'}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous --model-name '{alias}'. Matches multiple models: "
            f"{', '.join(matches)}"
        )
    return matches[0]


def phase_status(state: dict[str, Any]) -> dict[str, bool]:
    raw_list = state.get("raw", [])
    dockerfiles = state.get("dockerfiles", [])
    restore_list = state.get("restore", [])
    pull_llm = bool(raw_list) and all(
        raw.get("downloaded") and raw.get("split") for raw in raw_list
    )
    push_docker = bool(dockerfiles) and all(
        entry.get("built") and entry.get("pushed") for entry in dockerfiles
    )
    if "publication" in state:
        push_docker = (
            push_docker and bool(state["publication"].get("complete"))
            and all(entry.get("digest") for entry in dockerfiles)
        )
    pull_dockerhub = bool(dockerfiles) and all(
        entry.get("pulled") for entry in dockerfiles
    )
    restored_names = {
        item.get("raw_filename") for item in restore_list if item.get("restored")
    }
    restore_llm = bool(raw_list) and all(
        raw.get("filename") in restored_names for raw in raw_list
    )
    return {
        "pull_llm": pull_llm,
        "push_docker": push_docker,
        "pull_dockerhub": pull_dockerhub,
        "restore_llm": restore_llm,
    }


def phase_progress(state: dict[str, Any]) -> dict[str, Any]:
    raw_list = state.get("raw", [])
    dockerfiles = state.get("dockerfiles", [])
    restore_list = state.get("restore", [])
    restored_names = {
        item.get("raw_filename") for item in restore_list if item.get("restored")
    }
    return {
        "raw": {
            "total": len(raw_list),
            "downloaded": sum(1 for raw in raw_list if raw.get("downloaded")),
            "split": sum(1 for raw in raw_list if raw.get("split")),
        },
        "images": {
            "total": len(dockerfiles),
            "built": sum(1 for entry in dockerfiles if entry.get("built")),
            "pushed": sum(1 for entry in dockerfiles if entry.get("pushed")),
            "pulled": sum(1 for entry in dockerfiles if entry.get("pulled")),
            "extracted": sum(1 for entry in dockerfiles if entry.get("extracted")),
        },
        "restore": {
            "total": len(raw_list),
            "restored": sum(
                1 for raw in raw_list if raw.get("filename") in restored_names
            ),
        },
    }


def is_pull_push_complete(state: dict[str, Any]) -> bool:
    status = phase_status(state)
    return status["pull_llm"] and status["push_docker"]


def resolve_model_names(
    args: argparse.Namespace,
    phase: str,
    require_complete: bool = False,
    base_dir: Path | None = None,
) -> list[str]:
    root = base_dir or BASE_DIR
    aliases = get_model_names(args)
    if not aliases:
        inferred = infer_single_model_name(root)
        if inferred:
            aliases = [inferred]
        else:
            raise ValueError(
                f"--model-name is required for {phase} unless exactly one "
                f"model_weights/*/{STATE_FILENAME} exists"
            )
    resolved = [resolve_model_alias(alias, root) for alias in aliases]
    if require_complete:
        for name in resolved:
            state = load_state(root / name)
            if not is_pull_push_complete(state):
                raise ValueError(
                    f"Model '{name}' is not ready for {phase}: "
                    f"expected pull_llm and push_docker to be complete "
                    f"(every raw downloaded and split, every dockerfile built and pushed)"
                )
    return resolved


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
