from __future__ import annotations

import argparse
import os
import sys

from src.constants import (
    DEFAULT_DOCKER_NAMESPACE,
    DEFAULT_MAX_DOCKER_IMAGE_SIZE,
    DEFAULT_MAX_PART_SIZE,
    DOCKER_PULL_PREFIX_ENV,
    PHASE_PULL_LLM,
    VALID_PHASES,
)
from src.huggingface import parse_huggingface_url
from src.naming import safe_path_name
from src.phases import PHASE_HANDLERS
from src.sizes import parse_size

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
        required=True,
        help=f"Comma-separated phases: {','.join(VALID_PHASES)}",
    )
    parser.add_argument("--huggingface-link")
    parser.add_argument("--model-name")
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
    if PHASE_PULL_LLM in args.phases and not args.model_name and args.huggingface_link:
        repo_id, _ = parse_huggingface_url(args.huggingface_link, args.revision)
        args.model_name = safe_path_name(repo_id.split("/", 1)[1])

    for phase in args.phases:
        print(f"=== {phase} ===", flush=True)
        PHASE_HANDLERS[phase](args)
    return 0
