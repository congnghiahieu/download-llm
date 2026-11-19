from __future__ import annotations

import argparse
import json
import logging
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
from src.run_logging import log_file, logger, run_logging
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
    parser.add_argument("--plan", action="store_true", help="Preview phase execution without writing data.")
    parser.add_argument(
        "--output",
        choices=("table", "json"),
        default=None,
        help="Output format for --list-models or --plan. Defaults to table.",
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
    parser.add_argument("--transfer-models", action="store_true",
                        help="Transfer restored model payloads over SSH, independently of phases.")
    parser.add_argument("--ssh-host", help="SSH config alias or username@hostname.")
    parser.add_argument("--ssh-option", action="append", default=[],
                        help="Repeatable OpenSSH option, e.g. Port=2222 or ProxyJump=bastion.")
    parser.add_argument("--destination-path", help="Absolute destination root; model folders are added beneath it.")
    parser.add_argument("--existing-destination", choices=("ask", "merge"),
                        help="Non-empty model folders: ask (default) or explicitly allow merge.")
    parser.add_argument("--dry-run", action="store_true", help="Preview transfer without remote payload writes.")
    args = parser.parse_args(argv)

    if args.transfer_models:
        if args.phases or args.list_models or args.plan:
            parser.error("--transfer-models cannot be combined with --phases, --list-models or --plan")
        if not args.ssh_host or not args.destination_path:
            parser.error("--transfer-models requires --ssh-host and --destination-path")
    elif (args.ssh_host or args.ssh_option or args.destination_path
          or args.existing_destination or args.dry_run):
        parser.error("SSH, destination and dry-run options require --transfer-models")
    args.existing_destination = args.existing_destination or "ask"

    if args.list_models and args.phases:
        parser.error("--list-models cannot be combined with --phases")
    if args.plan and (not args.phases or args.list_models):
        parser.error("--plan requires --phases and cannot be combined with --list-models")
    if not args.list_models and not args.phases and not args.transfer_models:
        parser.error("--phases is required unless --list-models or --transfer-models is set")
    if args.output is not None and not (args.list_models or args.plan):
        parser.error("--output requires --list-models or --plan")
    if args.output is None:
        args.output = "table"
    phases = [
        phase.strip() for phase in (args.phases or "").split(",") if phase.strip()
    ]
    invalid = [phase for phase in phases if phase not in VALID_PHASES]
    if invalid:
        parser.error(f"Invalid phases: {', '.join(invalid)}")
    if args.plan and not phases:
        parser.error("--plan requires at least one phase")
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
    if getattr(args, "list_models", False):
        from src.listing import run_list_models

        return run_list_models(args)
    if getattr(args, "plan", False):
        from src.planning import make_plan, render_plan

        render_plan(make_plan(args), args.output)
        return 0

    with run_logging():
        logger.info("Run started")
        try:
            status = _execute_work(args)
        except KeyboardInterrupt:
            logger.warning("Run interrupted")
            raise
        except Exception as exc:
            logger.error("Run failed (%s)", type(exc).__name__)
            log_file(logging.ERROR, "Run failure traceback", exc_info=True)
            raise
        logger.log(
            logging.INFO if status == 0 else logging.ERROR,
            "Run finished (exit=%d)",
            status,
        )
        return status


def _execute_work(args: argparse.Namespace) -> int:
    if getattr(args, "transfer_models", False):
        from src.transfer import run_transfer

        return run_transfer(args)
    from src.pipeline import run_models
    from src.planning import make_plan, render_plan

    validate_execution_prereqs(args)
    plan = make_plan(args, allow_sudo=True)
    if getattr(args, "interactive_plan", False):
        from src.interactive import preview_plan

        if not preview_plan(plan):
            return 0
    else:
        render_plan(plan, to_stderr=True)
    log_file(logging.INFO, "Execution plan: %s", json.dumps(plan.output()))
    for role, storage in plan.storage.items():
        if storage.measurement_error:
            log_file(logging.INFO, "Storage measurement unavailable [role=%s, path=%s]: %s", role, storage.path, storage.measurement_error)
    authorized = plan.authorize_run()
    log_file(logging.INFO, "Execution authorized: %s", authorized)
    if not authorized:
        return 1
    run_models(plan.models, plan.max_workers, plan.scheduler())
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from src.interactive import run_interactive

        return run_interactive()
    args = parse_args(argv)
    return execute(args)
