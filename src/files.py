from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from src.constants import (
    BUFFER_SIZE,
    EXTRACTED_DIRNAME,
    PARTS_DIRNAME,
    RESTORED_DIRNAME,
)
from src.naming import part_label
from src.sizes import human_size


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


def path_matches(path: Path, size_bytes: int | None, sha256: str | None) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if size_bytes is not None and path.stat().st_size != size_bytes:
        return False
    return not (sha256 is not None and sha256_file(path) != sha256)


def part_files_exist(
    parts: list[dict[str, Any]], model_dir: Path | None = None
) -> bool:
    for part in parts:
        stored = Path(str(part["path"]))
        candidate = stored if stored.is_absolute() else None
        if candidate is None and model_dir is not None:
            candidate = model_dir / stored
        elif candidate is None:
            candidate = stored
        if not path_matches(
            candidate, int(part["size_bytes"]), part.get("sha256")
        ):
            if model_dir is not None and not stored.is_absolute():
                legacy = stored
                if legacy.exists() and path_matches(
                    legacy, int(part["size_bytes"]), part.get("sha256")
                ):
                    continue
            return False
    return True


def extracted_parts_exist(model_dir: Path, parts: list[dict[str, Any]]) -> bool:
    for part in parts:
        part_path = model_dir / EXTRACTED_DIRNAME / PARTS_DIRNAME / part["part_filename"]
        if not path_matches(part_path, int(part["size_bytes"]), part.get("sha256")):
            return False
    return True


def restored_file_valid(model_dir: Path, raw: dict[str, Any]) -> bool:
    restored_path = model_dir / EXTRACTED_DIRNAME / RESTORED_DIRNAME / raw["filename"]
    return path_matches(
        restored_path,
        int(raw["size_bytes"]) if "size_bytes" in raw else None,
        raw.get("sha256"),
    )


def split_raw_file(
    raw_path: Path,
    raw_rel: Path,
    parts_dir: Path,
    max_part_size: int,
    progress_cb: Any | None = None,
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
                    if progress_cb is not None:
                        progress_cb(len(data))

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
                    "path": f"{PARTS_DIRNAME}/{part_rel.as_posix()}",
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


def restore_raw_file(
    model_dir: Path,
    raw: dict[str, Any],
    parts: list[dict[str, Any]],
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    restored_path = model_dir / EXTRACTED_DIRNAME / RESTORED_DIRNAME / raw["filename"]
    restored_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with restored_path.open("wb") as target:
        for part in sorted(parts, key=lambda item: int(item["index"])):
            part_path = (
                model_dir / EXTRACTED_DIRNAME / PARTS_DIRNAME / part["part_filename"]
            )
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
                    if progress_cb is not None:
                        progress_cb(len(chunk))
            part["extracted"] = True

    restored_sha = digest.hexdigest()
    if written != raw["size_bytes"] or restored_sha != raw["sha256"]:
        raise ValueError(f"Restored file mismatch for {raw['filename']}")
    return {
        "raw_filename": raw["filename"],
        "restored_path": f"{EXTRACTED_DIRNAME}/{RESTORED_DIRNAME}/{Path(str(raw['filename'])).as_posix()}",
        "size_bytes": written,
        "size": human_size(written),
        "sha256": restored_sha,
        "restored": True,
    }
