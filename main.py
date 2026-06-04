from __future__ import annotations

from src.cli import main, parse_args
from src.constants import (
    ALPINE_IMAGE_NAME,
    BASE_DIR,
    BUFFER_SIZE,
    DEFAULT_MAX_DOCKER_IMAGE_SIZE,
    DEFAULT_MAX_PART_SIZE,
    VALID_PHASES,
)
from src.docker import (
    confirm_unpushed,
    dockerfile_text,
    ensure_dockerfiles,
    extract_image_parts,
    group_parts,
    run_docker,
)
from src.files import remove_file, restore_raw_file, sha256_file, split_raw_file
from src.huggingface import parse_huggingface_url
from src.naming import (
    docker_pull_tag,
    docker_repo_name,
    join_docker_prefix,
    part_label,
    safe_path_name,
)
from src.phases import (
    PHASE_HANDLERS,
    phase_pull_dockerhub,
    phase_pull_llm,
    phase_push_docker,
    phase_restore_llm,
)
from src.sizes import human_size, parse_size
from src.state import (
    atomic_write_json,
    infer_single_model_name,
    load_state,
    make_initial_state,
    resolve_model_name,
    save_state,
)

__all__ = [
    "ALPINE_IMAGE_NAME",
    "BASE_DIR",
    "BUFFER_SIZE",
    "DEFAULT_MAX_DOCKER_IMAGE_SIZE",
    "DEFAULT_MAX_PART_SIZE",
    "PHASE_HANDLERS",
    "VALID_PHASES",
    "atomic_write_json",
    "confirm_unpushed",
    "docker_pull_tag",
    "docker_repo_name",
    "dockerfile_text",
    "ensure_dockerfiles",
    "extract_image_parts",
    "group_parts",
    "human_size",
    "infer_single_model_name",
    "join_docker_prefix",
    "load_state",
    "main",
    "make_initial_state",
    "parse_args",
    "parse_huggingface_url",
    "parse_size",
    "part_label",
    "phase_pull_dockerhub",
    "phase_pull_llm",
    "phase_push_docker",
    "phase_restore_llm",
    "remove_file",
    "resolve_model_name",
    "restore_raw_file",
    "run_docker",
    "safe_path_name",
    "save_state",
    "sha256_file",
    "split_raw_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
