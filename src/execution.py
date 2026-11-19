from __future__ import annotations

import argparse
import os
import sys
import time

from src.constants import BASE_DIR, PHASE_PULL_LLM
from src.huggingface import parse_huggingface_url
from src.naming import safe_path_name
from src.phases import PHASE_HANDLERS
from src.run_logging import logger


def resolve_pull_targets(
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    from src.state import get_huggingface_links, get_model_names

    links = get_huggingface_links(args)
    if not links:
        raise ValueError(f"--huggingface-link is required for {PHASE_PULL_LLM}")
    seen: dict[str, None] = {}
    for link in links:
        seen.setdefault(link, None)
    unique_links = list(seen.keys())

    explicit_names = get_model_names(args)
    if len(unique_links) > 1 and explicit_names:
        print(
            "WARNING: --model-name is ignored with multiple --huggingface-link values; "
            "model names are derived from each link.",
            file=sys.stderr,
            flush=True,
        )
    targets: list[tuple[str, str]] = []
    for link in unique_links:
        repo_id, _ = parse_huggingface_url(link, getattr(args, "revision", None))
        if len(unique_links) == 1 and explicit_names:
            model_name = safe_path_name(explicit_names[0])
        else:
            model_name = safe_path_name(repo_id.split("/", 1)[1])
        targets.append((link, model_name))
    return targets


def copy_args_for_model(
    args: argparse.Namespace,
    model_name: str,
    huggingface_link: str | None = None,
) -> argparse.Namespace:
    single = argparse.Namespace(**vars(args))
    single.model_name = model_name
    single.model_names = [model_name]
    if huggingface_link is not None:
        single.huggingface_link = huggingface_link
        single.huggingface_links = [huggingface_link]
    return single


def run_phases_for_model(args: argparse.Namespace) -> None:
    label = args.model_name or ""
    for phase in args.phases:
        start = time.monotonic()
        logger.info("Phase %s started [model=%r]", phase, label)
        try:
            PHASE_HANDLERS[phase](args)
        except Exception as exc:
            logger.error(
                "Phase %s failed [model=%r] (%s)", phase, label, type(exc).__name__
            )
            raise
        logger.info(
            "Phase %s finished [model=%r] (%.2fs)",
            phase,
            label,
            time.monotonic() - start,
        )


def detect_command_prefix() -> str:
    executable = sys.executable or ""
    if (
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("UV_PROJECT_ENVIRONMENT")
        or ".venv" in executable
    ):
        return "uv run python main.py"
    return "python3 main.py"


def build_retry_command(item: argparse.Namespace) -> str:
    import shlex

    from src.state import get_huggingface_links

    parts = [detect_command_prefix()]
    parts.append(f"--phases {shlex.quote(','.join(item.phases))}")
    links = get_huggingface_links(item)
    if links:
        for link in links:
            parts.append(f"--huggingface-link {shlex.quote(link)}")
    if getattr(item, "model_name", None):
        parts.append(f"--model-name {shlex.quote(item.model_name)}")
    if getattr(item, "revision", None):
        parts.append(f"--revision {shlex.quote(item.revision)}")
    if getattr(item, "docker_namespace", None):
        parts.append(f"--docker-namespace {shlex.quote(item.docker_namespace)}")
    if getattr(item, "docker_pull_prefix", None):
        parts.append(f"--docker-pull-prefix {shlex.quote(item.docker_pull_prefix)}")
    if getattr(item, "max_part_size", None):
        parts.append(f"--max-part-size {shlex.quote(item.max_part_size)}")
    if getattr(item, "max_docker_image_size", None):
        parts.append(
            f"--max-docker-image-size {shlex.quote(item.max_docker_image_size)}"
        )
    weights_dir = getattr(item, "model_weights_dir", None)
    if weights_dir and not _is_default_weights_dir(str(weights_dir)):
        parts.append(f"--model-weights-dir {shlex.quote(str(weights_dir))}")
    for flag in ("ssh_host", "destination_path"):
        if value := getattr(item, flag, None):
            parts.append(f"--{flag.replace('_', '-')} {shlex.quote(value)}")
    for option in getattr(item, "ssh_option", []):
        parts.append(f"--ssh-option {shlex.quote(option)}")
    for flag in ("keep_raw", "keep_parts", "keep_images", "keep_extracted_parts", "keep_local"):
        if getattr(item, flag, False):
            parts.append(f"--{flag.replace('_', '-')}")
    return " ".join(parts)


def _is_default_weights_dir(weights_dir: str) -> bool:
    from src.state import normalize_path

    try:
        current = normalize_path(weights_dir)
    except OSError:
        return False
    try:
        default = normalize_path(BASE_DIR)
    except OSError:
        return False
    return current == default


def report_multi_failures(
    failures: dict[str, BaseException],
    retry_commands: dict[str, str],
    total: int,
) -> None:
    print(f"FAILED {len(failures)}/{total} model(s):", flush=True)
    for label in sorted(failures):
        exc = failures[label]
        print(
            f"FAILED [{label}]: {type(exc).__name__}: {exc}",
            flush=True,
        )
    print("Retry each failed model individually:", flush=True)
    for label in sorted(failures):
        print(f"  {retry_commands[label]}", flush=True)
    print(
        "Or rerun the original command: completed files and entries "
        "are skipped via state.json, only unfinished work resumes.",
        flush=True,
    )
    summary = ", ".join(sorted(failures.keys()))
    raise RuntimeError(f"{len(failures)} model(s) failed: {summary}")


def run_parallel_models(
    work_items: list[argparse.Namespace], args: argparse.Namespace
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = getattr(args, "max_workers", None) or len(work_items)
    max_workers = max(1, min(max_workers, len(work_items)))
    print(
        f"Running {len(work_items)} models in parallel with {max_workers} workers",
        flush=True,
    )
    retry_commands = {
        (item.model_name or ""): build_retry_command(item) for item in work_items
    }
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(run_phases_for_model, item): (item.model_name or "")
            for item in work_items
        }
        failures: dict[str, BaseException] = {}
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect per-model failures
                failures[label] = exc
                print(
                    f"FAILED [{label}]: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if failures:
            report_multi_failures(failures, retry_commands, len(work_items))


def run_sequential_models(
    work_items: list[argparse.Namespace], args: argparse.Namespace | None = None
) -> None:
    print(
        f"Running {len(work_items)} models sequentially in input order",
        flush=True,
    )
    retry_commands = {
        (item.model_name or ""): build_retry_command(item) for item in work_items
    }
    failures: dict[str, BaseException] = {}
    for item in work_items:
        label = item.model_name or ""
        try:
            run_phases_for_model(item)
        except Exception as exc:  # noqa: BLE001 - best effort, continue next model
            failures[label] = exc
            print(
                f"FAILED [{label}]: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if failures:
        report_multi_failures(failures, retry_commands, len(work_items))


def run_multi_models(
    work_items: list[argparse.Namespace], args: argparse.Namespace
) -> None:
    if getattr(args, "max_workers", None) == 1:
        run_sequential_models(work_items, args)
    else:
        run_parallel_models(work_items, args)
