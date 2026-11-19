from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from src.console import Confirm, Console, Panel, Prompt, Table
from src.constants import (
    BASE_DIR,
    DEFAULT_DOCKER_NAMESPACE,
    DEFAULT_MAX_DOCKER_IMAGE_SIZE,
    DEFAULT_MAX_PART_SIZE,
    DOCKER_PULL_PREFIX_ENV,
    MODEL_WEIGHTS_DIR_ENV,
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
)
from src.interactive_prompts import (
    FLOW_PULL_PUSH,
    FLOW_RESTORE,
    prompt_execution_mode,
    prompt_hf_token,
    prompt_keep_flags,
    prompt_links,
    prompt_model_names,
    prompt_output_format,
    prompt_phases,
    prompt_sizes,
    prompt_weights_dir,
    show_available_models,
)

if TYPE_CHECKING:
    from src.planning import PhasePlan

__all__ = [
    "FLOW_PULL_PUSH",
    "FLOW_RESTORE",
    "build_args",
    "prompt_execution_mode",
    "prompt_hf_token",
    "prompt_keep_flags",
    "prompt_links",
    "prompt_model_names",
    "prompt_output_format",
    "prompt_phases",
    "prompt_sizes",
    "prompt_weights_dir",
    "run_interactive",
    "show_available_models",
    "show_summary",
]


def show_summary(
    console: Console,
    phases: list[str],
    links: list[str],
    models: list[str],
    revision: str | None,
    namespace: str | None,
    prefix: str | None,
    max_workers: int | None,
    hf_token_set: bool,
    weights_dir: str | None = None,
    max_part_size: str | None = None,
    max_docker_image_size: str | None = None,
    keep_flags: dict[str, bool] | None = None,
) -> None:
    table = Table(title="Selected configuration")
    table.add_column("Key")
    table.add_column("Value")
    if weights_dir is not None:
        table.add_row("model-weights-dir", weights_dir)
    table.add_row("phases", ",".join(phases))
    if links:
        table.add_row("huggingface-links", "\n".join(links))
    if models:
        table.add_row("model-names", "\n".join(models))
    if revision is not None:
        table.add_row("revision", revision)
    if namespace is not None:
        table.add_row("docker-namespace", namespace)
    if prefix is not None:
        table.add_row("docker-pull-prefix", prefix)
    if max_part_size is not None:
        table.add_row("max-part-size", max_part_size)
    if max_docker_image_size is not None:
        table.add_row("max-docker-image-size", max_docker_image_size)
    if keep_flags:
        enabled = [key for key, value in keep_flags.items() if value]
        table.add_row("keep-flags", ", ".join(enabled) if enabled else "none")
    if max_workers is None:
        table.add_row("execution", "parallel (default workers)")
    elif max_workers == 1:
        table.add_row("execution", "sequential")
    else:
        table.add_row("execution", f"parallel ({max_workers} workers)")
    table.add_row("HF_TOKEN", "set" if hf_token_set else "not set")
    console.print(table)


def build_args(
    phases: list[str],
    links: list[str],
    models: list[str],
    revision: str | None,
    namespace: str,
    prefix: str | None,
    max_workers: int | None,
    weights_dir: str | None = None,
    max_part_size: str | None = None,
    max_docker_image_size: str | None = None,
    keep_flags: dict[str, bool] | None = None,
) -> argparse.Namespace:
    from src.sizes import parse_size

    part_size = max_part_size or DEFAULT_MAX_PART_SIZE
    image_size = max_docker_image_size or DEFAULT_MAX_DOCKER_IMAGE_SIZE
    flags = keep_flags or {}
    return argparse.Namespace(
        phases=phases,
        list_models=False,
        output="table",
        model_weights_dir=weights_dir
        or os.environ.get(MODEL_WEIGHTS_DIR_ENV, BASE_DIR.as_posix()),
        huggingface_links=list(links),
        huggingface_link=links[0] if links else None,
        model_names=list(models),
        model_name=models[0] if models else None,
        revision=revision,
        docker_namespace=namespace,
        docker_pull_prefix=prefix,
        max_part_size=part_size,
        max_docker_image_size=image_size,
        max_part_size_bytes=parse_size(part_size),
        max_docker_image_size_bytes=parse_size(image_size),
        max_workers=max_workers,
        keep_raw=bool(flags.get("keep_raw", False)),
        keep_parts=bool(flags.get("keep_parts", False)),
        keep_images=bool(flags.get("keep_images", False)),
        keep_extracted_parts=bool(flags.get("keep_extracted_parts", False)),
    )


def run_interactive() -> int:
    from src.cli import execute
    from src.listing import run_list_models

    console = Console()
    console.print(
        Panel(
            "Download, split, ship, and restore Hugging Face LLM weights "
            "via Docker Hub.",
            title="download-llm interactive",
        )
    )
    try:
        weights_dir = prompt_weights_dir(console)
        base_dir = Path(weights_dir)
        phases = prompt_phases(console)
        if phases == ["transfer"]:
            return prompt_transfer(console, weights_dir)
        if phases is None:
            output = prompt_output_format(console)
            return run_list_models(
                argparse.Namespace(
                    list_models=True,
                    output=output,
                    model_weights_dir=weights_dir,
                )
            )
        hf_token: str | None = None
        if PHASE_PULL_LLM in phases:
            hf_token = prompt_hf_token(console)
        revision: str | None = None
        if PHASE_PULL_LLM in phases:
            revision = Prompt.ask("Revision", default="main").strip() or "main"
        namespace = DEFAULT_DOCKER_NAMESPACE
        if PHASE_PUSH_DOCKER in phases:
            namespace = (
                Prompt.ask("Docker namespace", default=DEFAULT_DOCKER_NAMESPACE).strip()
                or DEFAULT_DOCKER_NAMESPACE
            )
        prefix: str | None = None
        if PHASE_PULL_DOCKERHUB in phases or PHASE_RESTORE_LLM in phases:
            console.print(
                "Optional DOCKER_PULL_PREFIX for pull via a proxy registry "
                "instead of direct Docker Hub access. Empty to skip."
            )
            default_prefix = os.environ.get(DOCKER_PULL_PREFIX_ENV) or ""
            entered = Prompt.ask("Docker pull prefix", default=default_prefix).strip()
            prefix = entered or None
        links: list[str] = []
        models: list[str] = []
        if PHASE_PULL_LLM in phases:
            links = prompt_links(console)
        else:
            require_complete = (
                PHASE_PULL_DOCKERHUB in phases or PHASE_RESTORE_LLM in phases
            ) and PHASE_PUSH_DOCKER not in phases
            models = prompt_model_names(console, require_complete, base_dir)
        count = len(links) if links else len(models)
        max_part_size, max_docker_image_size = prompt_sizes(console, phases)
        keep_flags = prompt_keep_flags(console, phases)
        max_workers = prompt_execution_mode(console, count)
        summary_part_size = max_part_size if PHASE_PULL_LLM in phases else None
        summary_image_size = (
            max_docker_image_size
            if (PHASE_PULL_LLM in phases or PHASE_PUSH_DOCKER in phases)
            else None
        )
        show_summary(
            console,
            phases,
            links,
            models,
            revision,
            namespace if PHASE_PUSH_DOCKER in phases else None,
            prefix,
            max_workers,
            hf_token is not None,
            weights_dir,
            summary_part_size,
            summary_image_size,
            keep_flags,
        )
        args = build_args(
            phases,
            links,
            models,
            revision,
            namespace,
            prefix,
            max_workers,
            weights_dir,
            max_part_size,
            max_docker_image_size,
            keep_flags,
        )
        args.interactive_plan = True
        return execute(args)
    except (KeyboardInterrupt, EOFError):
        console.print("\nAborted by user.")
        return 1


def preview_plan(plan: PhasePlan) -> bool:
    from src.planning import render_plan

    def limit(label: str, default: int) -> int:
        while True:
            value = Prompt.ask(label, default=str(default))
            if value.isdecimal() and int(value) > 0:
                return int(value)
            print("Enter a positive integer.")

    while True:
        render_plan(plan)
        action = Prompt.ask("Plan", choices=["run", "edit", "cancel"], default="run")
        if action != "edit":
            return action == "run"
        workers = limit("Max active models", plan.requested_workers)
        limits = {kind: limit(kind, current) for kind, current in plan.limits.items()}
        plan.configure(workers, limits)


def prompt_transfer(console: Console, weights_dir: str) -> int:
    from src.cli import execute, parse_args

    models = prompt_model_names(console, False, Path(weights_dir))
    host = Prompt.ask("SSH host (username@hostname or config alias)").strip()
    destination = Prompt.ask("Absolute destination path on server").strip()
    options = []
    while True:
        option = Prompt.ask("SSH option (Key=Value, empty to finish)", default="").strip()
        if not option:
            break
        options.extend(["--ssh-option", option])
    workers = prompt_execution_mode(console, len(models))
    policy = Prompt.ask("Existing destination policy", choices=["ask", "merge"], default="ask")
    dry_run = Confirm.ask("Dry run only?", default=False)
    argv = ["--transfer-models", "--model-weights-dir", weights_dir,
            "--ssh-host", host, "--destination-path", destination,
            "--existing-destination", policy, *options]
    for name in models:
        argv.extend(["--model-name", name])
    if workers is not None:
        argv.extend(["--max-workers", str(workers)])
    if dry_run:
        argv.append("--dry-run")
    args = parse_args(argv)
    console.print(f"Transfer {', '.join(models)} to {host}:{destination}/<model>/")
    if not Confirm.ask("Start now?", default=True):
        return 0
    return execute(args)
