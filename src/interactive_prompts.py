from __future__ import annotations

import os
from pathlib import Path

from src.console import Confirm, Console, IntPrompt, Prompt, Table
from src.constants import (
    BASE_DIR,
    DEFAULT_MAX_DOCKER_IMAGE_SIZE,
    DEFAULT_MAX_PART_SIZE,
    MODEL_WEIGHTS_DIR_ENV,
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
    VALID_PHASES,
)

FLOW_PULL_PUSH = f"{PHASE_PULL_LLM},{PHASE_PUSH_DOCKER}"
FLOW_RESTORE = f"{PHASE_PULL_DOCKERHUB},{PHASE_RESTORE_LLM}"


def prompt_weights_dir(console: Console) -> str:
    default = os.environ.get(MODEL_WEIGHTS_DIR_ENV, BASE_DIR.as_posix())
    entered = Prompt.ask("Model weights dir", default=default).strip()
    return entered or default


def prompt_output_format(console: Console) -> str:
    return Prompt.ask("Output format", choices=["table", "json"], default="table")


def prompt_phases(console: Console) -> list[str] | None:
    table = Table(title="Select workflow")
    table.add_column("Choice", justify="center")
    table.add_column("Phases")
    table.add_column("Use case")
    table.add_row("1", FLOW_PULL_PUSH, "Download from HF and ship via Docker Hub")
    table.add_row("2", FLOW_RESTORE, "Pull from Docker Hub and restore files")
    table.add_row("3", "custom", "Enter phases manually, single phase allowed")
    table.add_row("4", "list", "Show models with phase status and sizes")
    table.add_row("5", "transfer", "Transfer restored models over SSH")
    table.add_row("6", "cleanup", "Clean up selected local model data")
    console.print(table)
    choice = Prompt.ask("Choose workflow", choices=["1", "2", "3", "4", "5", "6"], default="1")
    if choice == "6":
        return ["cleanup"]
    if choice == "5":
        return ["transfer"]
    if choice == "1":
        return [PHASE_PULL_LLM, PHASE_PUSH_DOCKER]
    if choice == "2":
        return [PHASE_PULL_DOCKERHUB, PHASE_RESTORE_LLM]
    if choice == "4":
        return None
    while True:
        raw = Prompt.ask(
            f"Enter phases, comma-separated from {','.join(VALID_PHASES)}"
        ).strip()
        phases = [part.strip() for part in raw.split(",") if part.strip()]
        invalid = [phase for phase in phases if phase not in VALID_PHASES]
        if not phases:
            console.print("[red]At least one phase is required.[/red]")
            continue
        if invalid:
            console.print(f"[red]Invalid phases: {', '.join(invalid)}[/red]")
            continue
        return phases


def prompt_hf_token(console: Console) -> str | None:
    console.print(
        "Optional: enter HF_TOKEN for faster pull_llm, "
        "especially for gated models. Empty to skip."
    )
    token = Prompt.ask("HF_TOKEN", password=True, default="").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        console.print("[green]HF_TOKEN set for this session.[/green]")
        return token
    return None


def prompt_links(console: Console) -> list[str]:
    from src.huggingface import parse_huggingface_url

    console.print(
        "Enter Hugging Face links. One prompt per step, "
        "comma-separated allowed. Empty to finish."
    )
    links: list[str] = []
    while True:
        label = f"Link #{len(links) + 1}"
        suffix = " (required)" if not links else " (empty to finish)"
        raw = Prompt.ask(label + suffix, default="").strip()
        if not raw:
            if not links:
                console.print("[red]At least one link is required.[/red]")
                continue
            break
        for part in [item.strip() for item in raw.split(",") if item.strip()]:
            try:
                parse_huggingface_url(part, None)
            except ValueError as exc:
                console.print(f"[red]Invalid link: {part} ({exc})[/red]")
                continue
            if part not in links:
                links.append(part)
    return links


def show_available_models(console: Console, base_dir: Path | None = None) -> None:
    from src.state import is_pull_push_complete, list_model_dirs, load_state

    table = Table(title="Available models in model_weights")
    table.add_column("Folder")
    table.add_column("repo_id")
    table.add_column("Ready")
    models = list_model_dirs(base_dir)
    if not models:
        console.print("[yellow]No model_weights/*/state.json found yet.[/yellow]")
        return
    for model_dir in models:
        try:
            state = load_state(model_dir)
            ready = "yes" if is_pull_push_complete(state) else "no"
            table.add_row(model_dir.name, str(state.get("repo_id", "")), ready)
        except Exception:  # noqa: BLE001 - invalid state must not fail list
            table.add_row(model_dir.name, "", "invalid")
    console.print(table)


def prompt_model_names(
    console: Console, require_complete: bool, base_dir: Path | None = None
) -> list[str]:
    from src.state import (
        effective_base_dir,
        is_pull_push_complete,
        load_state,
        resolve_model_alias,
    )

    root = base_dir or effective_base_dir()
    show_available_models(console, root)
    console.print(
        "Enter model names or aliases. Accepts model_name, folder, "
        "docker_repo_name, repo_id, or url. "
        "Comma-separated allowed. Empty to finish."
    )
    names: list[str] = []
    while True:
        label = f"Model #{len(names) + 1}"
        suffix = " (required)" if not names else " (empty to finish)"
        raw = Prompt.ask(label + suffix, default="").strip()
        if not raw:
            if not names:
                console.print("[red]At least one model is required.[/red]")
                continue
            break
        for part in [item.strip() for item in raw.split(",") if item.strip()]:
            try:
                resolved = resolve_model_alias(part, root)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            if require_complete:
                try:
                    state = load_state(root / resolved)
                except Exception as exc:  # noqa: BLE001 - report invalid, keep prompting
                    console.print(f"[red]{exc}[/red]")
                    continue
                if not is_pull_push_complete(state):
                    console.print(
                        f"[red]Model '{resolved}' is not ready: expected "
                        "pull_llm and push_docker complete.[/red]"
                    )
                    continue
            if resolved not in names:
                names.append(resolved)
    return names


def prompt_execution_mode(console: Console, count: int) -> int | None:
    if count <= 1:
        return None
    console.print(
        "Multiple targets detected. Parallel runs pipelines concurrently, "
        "sequential runs them one by one in input order."
    )
    mode = Prompt.ask(
        "Execution mode", choices=["parallel", "sequential"], default="parallel"
    )
    if mode == "sequential":
        return 1
    workers = IntPrompt.ask("Max workers", default=count)
    if workers < 1:
        console.print("[yellow]Max workers must be >= 1, using 1.[/yellow]")
        return 1
    return min(workers, count)


def prompt_sizes(console: Console, phases: list[str]) -> tuple[str, str]:
    from src.sizes import parse_size

    max_part = DEFAULT_MAX_PART_SIZE
    max_image = DEFAULT_MAX_DOCKER_IMAGE_SIZE
    if PHASE_PULL_LLM in phases:
        while True:
            entered = Prompt.ask("Max part size", default=max_part).strip() or max_part
            try:
                parse_size(entered)
                max_part = entered
                break
            except Exception as exc:  # noqa: BLE001 - re-prompt on invalid size
                console.print(f"[red]Invalid size: {exc}[/red]")
    if PHASE_PULL_LLM in phases or PHASE_PUSH_DOCKER in phases:
        while True:
            entered = (
                Prompt.ask("Max docker image size", default=max_image).strip()
                or max_image
            )
            try:
                parse_size(entered)
                max_image = entered
                break
            except Exception as exc:  # noqa: BLE001 - re-prompt on invalid size
                console.print(f"[red]Invalid size: {exc}[/red]")
    return max_part, max_image


def prompt_keep_flags(console: Console, phases: list[str]) -> dict[str, bool]:
    flags = {
        "keep_raw": False,
        "keep_parts": False,
        "keep_images": False,
        "keep_extracted_parts": False,
    }
    if PHASE_PULL_LLM in phases:
        flags["keep_raw"] = Confirm.ask("Keep raw files after split?", default=False)
    if PHASE_PUSH_DOCKER in phases:
        flags["keep_parts"] = Confirm.ask(
            "Keep part files after push?", default=False
        )
    if PHASE_PUSH_DOCKER in phases or PHASE_RESTORE_LLM in phases:
        flags["keep_images"] = Confirm.ask(
            "Keep docker images locally?", default=False
        )
    if PHASE_RESTORE_LLM in phases:
        flags["keep_extracted_parts"] = Confirm.ask(
            "Keep extracted parts after restore?", default=False
        )
    return flags
