from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download

BASE_DIR = Path("model_weights")
DEFAULT_MAX_PART_SIZE = "5GB"
DEFAULT_MAX_DOCKER_IMAGE_SIZE = "5GB"
BUFFER_SIZE = 8 * 1024 * 1024
VALID_PHASES = ("pull_llm", "push_docker", "pull_dockerhub", "restore_llm")


def parse_size(value: str) -> int:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*", value, re.IGNORECASE
    )
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    factors = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    if unit not in factors:
        raise argparse.ArgumentTypeError(f"Invalid size unit: {unit}")
    size = int(number * factors[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError("Size must be greater than zero")
    return size


def human_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f}{unit}" if unit != "B" else f"{size}B"
        value /= 1000
    return f"{size}B"


def parse_huggingface_url(url: str, revision: str | None) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        raise ValueError("Hugging Face URL must use huggingface.co")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face URL must include owner and repo name")
    repo_id = "/".join(parts[:2])
    url_revision = None
    if len(parts) >= 4 and parts[2] == "tree":
        url_revision = "/".join(parts[3:])
    return repo_id, revision or url_revision or "main"


def safe_path_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return safe or "model"


def docker_repo_name(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-._")
    return safe or "model"


def part_label(index: int) -> str:
    return f"part{index:04d}"


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
    state_path = model_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with state_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_single_model_name() -> str | None:
    if not BASE_DIR.exists():
        return None
    candidates = [
        path.name for path in BASE_DIR.iterdir() if (path / "state.json").exists()
    ]
    return candidates[0] if len(candidates) == 1 else None


def resolve_model_name(args: argparse.Namespace, phase: str) -> str:
    if args.model_name:
        return safe_path_name(args.model_name)
    inferred = infer_single_model_name()
    if inferred:
        return inferred
    raise ValueError(
        f"--model-name is required for {phase} unless exactly one model_weights/*/state.json exists"
    )


def save_state(model_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(model_dir / "state.json", state)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def remove_file(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


def run_docker(args: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(["docker", *args]), flush=True)
    return subprocess.run(["docker", *args], check=True, text=True)


def split_raw_file(
    raw_path: Path,
    raw_rel: Path,
    parts_dir: Path,
    max_part_size: int,
) -> tuple[str, list[dict[str, Any]]]:
    raw_digest = hashlib.sha256()
    part_entries: list[dict[str, Any]] = []
    suffix = raw_path.suffix
    stem = raw_path.stem
    part_index = 1

    with raw_path.open("rb") as source:
        while True:
            first = source.read(min(BUFFER_SIZE, max_part_size))
            if not first:
                break

            part_rel = raw_rel.parent / f"{stem}-{part_label(part_index)}{suffix}"
            part_path = parts_dir / part_rel
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_digest = hashlib.sha256()
            written = 0

            with part_path.open("wb") as target:
                chunk = first
                while chunk:
                    remaining = max_part_size - written
                    data = chunk[:remaining]
                    extra = chunk[remaining:]
                    target.write(data)
                    written += len(data)
                    raw_digest.update(data)
                    part_digest.update(data)

                    if written >= max_part_size:
                        if extra:
                            source.seek(-len(extra), os.SEEK_CUR)
                        break
                    chunk = source.read(min(BUFFER_SIZE, max_part_size - written))

            part_entries.append(
                {
                    "raw_filename": raw_rel.as_posix(),
                    "index": part_index,
                    "part_filename": part_rel.as_posix(),
                    "path": (parts_dir / part_rel).as_posix(),
                    "size_bytes": written,
                    "size": human_size(written),
                    "sha256": part_digest.hexdigest(),
                    "deleted": False,
                    "extracted": False,
                    "restored": False,
                }
            )
            part_index += 1

    return raw_digest.hexdigest(), part_entries


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


def phase_pull_llm(args: argparse.Namespace) -> None:
    if not args.huggingface_link:
        raise ValueError("--huggingface-link is required for pull_llm")

    repo_id, revision = parse_huggingface_url(args.huggingface_link, args.revision)
    model_name = safe_path_name(args.model_name or repo_id.split("/", 1)[1])
    model_dir = BASE_DIR / model_name
    raw_dir = model_dir / "raw"
    parts_dir = model_dir / "parts"
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

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    for filename in files:
        print(f"Downloading {filename}", flush=True)
        raw_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="model",
                local_dir=raw_dir,
            )
        )
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
    lines = ["FROM alpine:3.22.4"]
    for part in group:
        part_path = part["part_filename"]
        lines.append(f"COPY parts/{part_path} /model_parts/{part_path}")
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
    dockerfiles_dir = model_dir / "dockerfiles"
    dockerfiles_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    repo = state.get("docker_repo_name") or docker_repo_name(state["model_name"])

    for index, group in enumerate(groups, start=1):
        label = part_label(index)
        dockerfile = dockerfiles_dir / f"Dockerfile.{label}"
        dockerfile.write_text(dockerfile_text(group), encoding="utf-8")
        entries.append(
            {
                "index": index,
                "label": label,
                "dockerfile": dockerfile.as_posix(),
                "tag": f"{namespace}/{repo}:{label}",
                "included_parts": [part["part_filename"] for part in group],
                "built": False,
                "pushed": False,
                "pulled": False,
                "removed": False,
                "extracted": False,
            }
        )
    state["dockerfiles"] = entries


def phase_push_docker(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, "push_docker")
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


def confirm_unpushed(tag: str, decision: str | None) -> tuple[bool, str | None]:
    if decision == "yes_all":
        return True, decision
    if decision == "no_all":
        return False, decision
    while True:
        print(f"ERROR: {tag} has pushed=false in state.json", file=sys.stderr)
        answer = input("Pull anyway? [y/n/ya/no]: ").strip().lower()
        if answer == "y":
            return True, decision
        if answer == "n":
            return False, decision
        if answer == "ya":
            return True, "yes_all"
        if answer == "no":
            return False, "no_all"
        print("Use y, n, ya, or no.", file=sys.stderr)


def phase_pull_dockerhub(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, "pull_dockerhub")
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)
    decision: str | None = None
    for entry in state.get("dockerfiles", []):
        if not entry.get("pushed"):
            should_pull, decision = confirm_unpushed(entry["tag"], decision)
            if not should_pull:
                continue
        if not entry.get("pulled"):
            run_docker(["pull", entry["tag"]])
            entry["pulled"] = True
            entry["removed"] = False
            save_state(model_dir, state)


def extract_image_parts(model_dir: Path, entry: dict[str, Any]) -> None:
    container_name = f"download-llm-{entry['label']}-{os.getpid()}"
    created = False
    try:
        run_docker(["create", "--name", container_name, entry["tag"]])
        created = True
        target_dir = model_dir / "extracted" / "parts"
        target_dir.mkdir(parents=True, exist_ok=True)
        run_docker(["cp", f"{container_name}:/model_parts/.", target_dir.as_posix()])
    finally:
        if created:
            subprocess.run(["docker", "rm", container_name], check=False, text=True)


def restore_raw_file(
    model_dir: Path, raw: dict[str, Any], parts: list[dict[str, Any]]
) -> dict[str, Any]:
    restored_path = model_dir / "extracted" / "restored" / raw["filename"]
    restored_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with restored_path.open("wb") as target:
        for part in sorted(parts, key=lambda item: int(item["index"])):
            part_path = model_dir / "extracted" / "parts" / part["part_filename"]
            if not part_path.exists():
                raise FileNotFoundError(f"Missing extracted part: {part_path}")
            actual_size = part_path.stat().st_size
            if actual_size != part["size_bytes"]:
                raise ValueError(f"Size mismatch for {part_path}")
            actual_sha = sha256_file(part_path)
            if actual_sha != part["sha256"]:
                raise ValueError(f"SHA256 mismatch for {part_path}")
            with part_path.open("rb") as source:
                while chunk := source.read(BUFFER_SIZE):
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            part["extracted"] = True

    restored_sha = digest.hexdigest()
    if written != raw["size_bytes"] or restored_sha != raw["sha256"]:
        raise ValueError(f"Restored file mismatch for {raw['filename']}")
    return {
        "raw_filename": raw["filename"],
        "restored_path": restored_path.as_posix(),
        "size_bytes": written,
        "size": human_size(written),
        "sha256": restored_sha,
        "restored": True,
    }


def phase_restore_llm(args: argparse.Namespace) -> None:
    model_name = resolve_model_name(args, "restore_llm")
    model_dir = BASE_DIR / model_name
    state = load_state(model_dir)

    for entry in state.get("dockerfiles", []):
        if not entry.get("pulled"):
            run_docker(["pull", entry["tag"]])
            entry["pulled"] = True
            save_state(model_dir, state)
        if not entry.get("extracted"):
            extract_image_parts(model_dir, entry)
            entry["extracted"] = True
            save_state(model_dir, state)
        if not args.keep_images and not entry.get("removed"):
            run_docker(["rmi", entry["tag"]])
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
                extracted = model_dir / "extracted" / "parts" / part["part_filename"]
                remove_file(extracted)
        save_state(model_dir, state)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, split, ship, and restore Hugging Face LLM weights via Docker Hub."
    )
    parser.add_argument(
        "--phases",
        required=True,
        help=f"Comma-separated phases: {','.join(VALID_PHASES)}",
    )
    parser.add_argument("--huggingface-link")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--docker-namespace", default="hieucien")
    parser.add_argument("--max-part-size", default=DEFAULT_MAX_PART_SIZE)
    parser.add_argument(
        "--max-docker-image-size", default=DEFAULT_MAX_DOCKER_IMAGE_SIZE
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--keep-extracted-parts", action="store_true")
    args = parser.parse_args(argv)

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    invalid = [phase for phase in phases if phase not in VALID_PHASES]
    if invalid:
        parser.error(f"Invalid phases: {', '.join(invalid)}")
    args.phases = phases
    args.max_part_size_bytes = parse_size(args.max_part_size)
    args.max_docker_image_size_bytes = parse_size(args.max_docker_image_size)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if "pull_llm" in args.phases and not args.model_name and args.huggingface_link:
        repo_id, _ = parse_huggingface_url(args.huggingface_link, args.revision)
        args.model_name = safe_path_name(repo_id.split("/", 1)[1])

    phase_handlers = {
        "pull_llm": phase_pull_llm,
        "push_docker": phase_push_docker,
        "pull_dockerhub": phase_pull_dockerhub,
        "restore_llm": phase_restore_llm,
    }
    for phase in args.phases:
        print(f"=== {phase} ===", flush=True)
        phase_handlers[phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
