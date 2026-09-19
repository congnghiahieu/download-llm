from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.console import RICH_AVAILABLE as _RICH_AVAILABLE
from src.console import Console, Table
from src.sizes import human_size
from src.state import (
    effective_base_dir,
    list_model_dirs,
    load_state,
    phase_progress,
    phase_status,
)

STATUS_DONE = "done"
STATUS_PENDING = "pending"
STATUS_INVALID = "invalid"


def summarize_model(model_dir: Path) -> dict[str, Any]:
    state = load_state(model_dir)
    status = phase_status(state)
    progress = phase_progress(state)
    raw_list = state.get("raw", [])
    dockerfiles = state.get("dockerfiles", [])
    raw_size_bytes = sum(int(raw.get("size_bytes", 0) or 0) for raw in raw_list)
    known_image_size = sum(
        int(entry["size_bytes"])
        for entry in dockerfiles
        if entry.get("size_bytes") is not None
    )
    unknown_images = sum(1 for entry in dockerfiles if entry.get("size_bytes") is None)
    image_size_bytes: int | None = known_image_size
    image_size = human_size(known_image_size)
    if dockerfiles and unknown_images == len(dockerfiles):
        image_size_bytes = None
        image_size = "unknown"
    return {
        "model": model_dir.name,
        "repo_id": str(state.get("repo_id", "")),
        "status": status,
        "progress": progress,
        "raw_files": len(raw_list),
        "raw_size_bytes": raw_size_bytes,
        "raw_size": human_size(raw_size_bytes),
        "images": len(dockerfiles),
        "image_size_bytes": image_size_bytes,
        "image_size": image_size,
        "unknown_images": unknown_images,
    }


def summarize_invalid(model_dir: Path, exc: BaseException) -> dict[str, Any]:
    return {
        "model": model_dir.name,
        "repo_id": "",
        "status": STATUS_INVALID,
        "progress": {
            "raw": {"total": 0, "downloaded": 0, "split": 0},
            "images": {
                "total": 0,
                "built": 0,
                "pushed": 0,
                "pulled": 0,
                "extracted": 0,
            },
            "restore": {"total": 0, "restored": 0},
        },
        "raw_files": 0,
        "raw_size_bytes": 0,
        "raw_size": "-",
        "images": 0,
        "image_size_bytes": None,
        "image_size": STATUS_INVALID,
        "unknown_images": 0,
        "error": f"Invalid state.json: {exc}",
    }


def format_image_size(summary: dict[str, Any]) -> str:
    if summary.get("status") == STATUS_INVALID:
        return STATUS_INVALID
    if not summary["images"]:
        return "-"
    if summary["unknown_images"] == 0:
        return str(summary["image_size"])
    if summary["image_size_bytes"]:
        return f"{summary['image_size']} (+{summary['unknown_images']} unknown)"
    return "unknown"


def _row_cells(summary: dict[str, Any]) -> list[str]:
    if summary.get("status") == STATUS_INVALID:
        return [
            summary["model"],
            STATUS_INVALID,
            STATUS_INVALID,
            STATUS_INVALID,
            STATUS_INVALID,
            "-",
            "-",
            "-",
            STATUS_INVALID,
        ]
    status = summary["status"]
    return [
        summary["model"],
        STATUS_DONE if status["pull_llm"] else STATUS_PENDING,
        STATUS_DONE if status["push_docker"] else STATUS_PENDING,
        STATUS_DONE if status["pull_dockerhub"] else STATUS_PENDING,
        STATUS_DONE if status["restore_llm"] else STATUS_PENDING,
        str(summary["raw_files"]),
        str(summary["raw_size"]),
        str(summary["images"]),
        format_image_size(summary),
    ]


def _render_plain_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "pull_llm",
        "push_docker",
        "pull_dockerhub",
        "restore_llm",
        "raw files",
        "raw size est",
        "images",
        "image size",
    ]
    rows = [_row_cells(summary) for summary in summaries]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("Models", flush=True)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)), flush=True)
    print("-+-".join("-" * width for width in widths), flush=True)
    for row in rows:
        print(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)), flush=True)


def render_table(summaries: list[dict[str, Any]]) -> None:
    if not _RICH_AVAILABLE:
        _render_plain_table(summaries)
        return
    console = Console()
    table = Table(title="Models")
    table.add_column("Model")
    table.add_column("pull_llm", justify="center")
    table.add_column("push_docker", justify="center")
    table.add_column("pull_dockerhub", justify="center")
    table.add_column("restore_llm", justify="center")
    table.add_column("raw files", justify="right")
    table.add_column("raw size est", justify="right")
    table.add_column("images", justify="right")
    table.add_column("image size", justify="right")
    for summary in summaries:
        table.add_row(*_row_cells(summary))
    console.print(table)


def run_list_models(args: argparse.Namespace) -> int:
    base_dir = effective_base_dir(args)
    output = getattr(args, "output", None) or "table"
    warn_console = Console(stderr=True)
    summaries: list[dict[str, Any]] = []
    for model_dir in list_model_dirs(base_dir):
        try:
            summaries.append(summarize_model(model_dir))
        except Exception as exc:  # noqa: BLE001 - invalid state must not fail list
            warn_console.print(
                f"[yellow]Invalid state.json for model "
                f"'{model_dir.name}': {exc}[/yellow]"
            )
            summaries.append(summarize_invalid(model_dir, exc))
    if output == "json":
        print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)
        return 0
    if not summaries:
        warn_console.print(
            f"[yellow]No models found in {base_dir.as_posix()}.[/yellow]"
        )
        return 0
    render_table(summaries)
    return 0
