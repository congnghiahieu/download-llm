from __future__ import annotations

import argparse
import os
import sys

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
    VALID_PHASES,
)
from src.docker import ensure_docker_available
from src.execution import (
    build_retry_command,
    copy_args_for_model,
    detect_command_prefix,
    report_multi_failures,
    resolve_pull_targets,
    run_multi_models,
    run_parallel_models,
    run_phases_for_model,
    run_sequential_models,
)
from src.phases import PHASE_HANDLERS
from src.sizes import parse_size

__all__ = [
    "DOCKER_PHASES",
    "build_retry_command",
    "copy_args_for_model",
    "detect_command_prefix",
    "execute",
    "main",
    "parse_args",
    "report_multi_failures",
    "resolve_pull_targets",
    "run_multi_models",
    "run_parallel_models",
    "run_phases_for_model",
    "run_sequential_models",
    "validate_execution_prereqs",
]

DOCKER_PHASES = (PHASE_PUSH_DOCKER, PHASE_PULL_DOCKERHUB, PHASE_RESTORE_LLM)

DESCRIPTION = (
    "Download, split, ship, and restore Hugging Face LLM weights via Docker Hub."
)
DOCKER_PULL_PREFIX_HELP = (
    "Optional registry/path prefix for pull/create/rmi, "
    "for example docker.internal/proxy-cache"
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--phases",
        required=False,
        default=None,
        help=f"Comma-separated phases: {','.join(VALID_PHASES)}",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models in the weights dir with 4-phase status and sizes.",
    )
    parser.add_argument(
        "--output",
        choices=("table", "json"),
        default=None,
        help="List output format. Defaults to table. Requires --list-models.",
    )
    parser.add_argument(
        "--model-weights-dir",
        default=os.environ.get(MODEL_WEIGHTS_DIR_ENV, BASE_DIR.as_posix()),
        help=(
            "Weights folder holding one subfolder per model. "
            f"Defaults to env {MODEL_WEIGHTS_DIR_ENV} or {BASE_DIR.as_posix()}."
        ),
    )
    parser.add_argument(
        "--huggingface-link",
        dest="huggingface_links",
        action="append",
        default=None,
        help="Hugging Face model URL. Repeat the flag or comma-separate for parallel pull.",
    )
    parser.add_argument(
        "--model-name",
        dest="model_names",
        action="append",
        default=None,
        help="Model name alias. Repeat the flag or comma-separate for parallel restore.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--docker-namespace", default=DEFAULT_DOCKER_NAMESPACE)
    parser.add_argument(
        "--docker-pull-prefix",
        default=os.environ.get(DOCKER_PULL_PREFIX_ENV),
        help=DOCKER_PULL_PREFIX_HELP,
    )
    parser.add_argument("--max-part-size", default=DEFAULT_MAX_PART_SIZE)
    parser.add_argument(
        "--max-docker-image-size", default=DEFAULT_MAX_DOCKER_IMAGE_SIZE
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Max parallel model workers. Defaults to number of models. "
            "Use --max-workers 1 for sequential execution in input order."
        ),
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--keep-extracted-parts", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models and args.phases:
        parser.error("--list-models cannot be combined with --phases")
    if not args.list_models and not args.phases:
        parser.error("--phases is required unless --list-models is set")
    if args.output is not None and not args.list_models:
        parser.error("--output requires --list-models")
    if args.output is None:
        args.output = "table"
    phases = [
        phase.strip() for phase in (args.phases or "").split(",") if phase.strip()
    ]
    invalid = [phase for phase in phases if phase not in VALID_PHASES]
    if invalid:
        parser.error(f"Invalid phases: {', '.join(invalid)}")
    args.phases = phases
    args.max_part_size_bytes = parse_size(args.max_part_size)
    args.max_docker_image_size_bytes = parse_size(args.max_docker_image_size)

    from src.state import get_huggingface_links, get_model_names

    args.huggingface_links = get_huggingface_links(args)
    args.model_names = get_model_names(args)
    args.huggingface_link = (
        args.huggingface_links[0] if args.huggingface_links else None
    )
    args.model_name = args.model_names[0] if args.model_names else None
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")
    return args


def validate_execution_prereqs(args: argparse.Namespace) -> None:
    if any(phase in DOCKER_PHASES for phase in args.phases):
        ensure_docker_available()


def execute(args: argparse.Namespace) -> int:
    from src.listing import run_list_models
    from src.state import effective_base_dir, resolve_model_names

    if getattr(args, "list_models", False):
        return run_list_models(args)
    validate_execution_prereqs(args)
    base_dir = effective_base_dir(args)
    if PHASE_PULL_LLM in args.phases:
        targets = resolve_pull_targets(args)
        if len(targets) == 1:
            link, model_name = targets[0]
            single = copy_args_for_model(args, model_name, link)
            for phase in single.phases:
                print(f"=== {phase} ===", flush=True)
                PHASE_HANDLERS[phase](single)
        else:
            work_items = [
                copy_args_for_model(args, model_name, link)
                for link, model_name in targets
            ]
            run_multi_models(work_items, args)
        return 0

    require_complete = any(
        phase in (PHASE_PULL_DOCKERHUB, PHASE_RESTORE_LLM) for phase in args.phases
    )
    model_names = resolve_model_names(
        args,
        ",".join(args.phases),
        require_complete=require_complete,
        base_dir=base_dir,
    )
    seen: dict[str, None] = {}
    for name in model_names:
        seen.setdefault(name, None)
    unique_names = list(seen.keys())
    if len(unique_names) == 1:
        args.model_name = unique_names[0]
        args.model_names = unique_names
        for phase in args.phases:
            print(f"=== {phase} ===", flush=True)
            PHASE_HANDLERS[phase](args)
    else:
        work_items = [copy_args_for_model(args, name) for name in unique_names]
        run_multi_models(work_items, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from src.interactive import run_interactive

        return run_interactive()
    args = parse_args(argv)
    return execute(args)
