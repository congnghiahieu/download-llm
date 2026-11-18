# AGENTS.md

This file guides coding agents working in this repository.

## What this repo does

1. This repo downloads Hugging Face weights, splits them into parts, ships the parts via Docker Hub, then pulls and restores the original files.
2. See `README.md` for common usage, multi-model execution, and interactive mode.

## Key modules

1. `main.py` re-exports the public CLI surface and dispatches to `src.cli:main`.
2. `src/cli.py` owns argument parsing, pull target resolution, sequential and parallel multi-model execution, and no-args interactive dispatch.
3. `src/interactive.py` owns the rich-based interactive prompts and builds an `argparse.Namespace` for `src.cli:execute`.
4. `src/state.py` owns `state.json` load and save, multi-value parsing, model alias resolution, pull plus push completeness checks, and strict 4-phase status via `phase_status`.
5. `src/phases.py` owns the four phase handlers `pull_llm`, `push_docker`, `pull_dockerhub`, and `restore_llm`.
6. `src/listing.py` owns `--list-models` summaries, rich table output, and JSON output.

## Environments

1. Develop on Python 3.10 with `uv` and full dependencies for a comfortable dev workflow.
2. Run internal and production on OS at least Ubuntu 22.04 or equivalent with system Python 3.10.
3. Expect old Docker equivalent to Ubuntu 22.04 in internal and production and newer Docker only in dev.
4. Assume `make` is available in every environment and keep Makefile as the stable interface.
5. Target Python 3.10 only with no Python 3.8 compatibility requirement.

## Packaging and dependencies

1. Keep the packaging goal as pure stdlib source runnable directly by `python3` without `.venv` or PyPI.
2. Keep dev comfortable with full dependencies via `uv` while runtime code must still run when optional dependencies are missing.
3. Make every runtime dependency optional with a plain text fallback following the `rich` pattern.
4. Prefer small and reputable dependencies and avoid adding a dependency when stdlib is adequate.
5. Do not use PyInstaller onefile for this repo.
6. Detect optional backends at runtime and use the best available one instead of hard-coding a default.
7. Cache stdlib Hugging Face downloads by ETag under `DOWNLOAD_LLM_CACHE` or `~/.cache/download-llm/hf` with resume and offline reuse.

## Critical conventions

1. Support repeated and comma-separated values for `--huggingface-link` and `--model-name`.
2. Treat `--max-workers 1` as sequential execution in input order and larger values as parallel execution.
3. Resolve `--model-name` against `model_name`, folder name, `docker_repo_name`, `repo_id`, and `url` in `model_weights/*/state.json`.
4. Require pull plus push completeness for `pull_dockerhub` and `restore_llm` and fail fast on invalid models.
5. Set `HF_TOKEN` only in the current process from interactive mode and never log token values.
6. Keep new dependencies small and reputable because this is a download tool.
7. Resolve the weights folder from `--model-weights-dir` or `MODEL_WEIGHTS_DIR` with `model_weights` default and thread it through every phase, list, and interactive flow.
8. Report 4-phase status from strict `state.json` flags only, with `done` or `pending` and no partial state.
9. Record docker image size into `state.json` at build time and backfill it when the image is still local, while older models without the field report `unknown`.
10. Serve `--list-models` as a standalone read-only flag with `table` or `json` output, and reject combining it with `--phases`.
11. Implement every CLI add, edit, or delete in both non-interactive mode (`src/cli.py`) and interactive mode (`src/interactive.py`) by default, and scope to one mode only on explicit instruction.

## Verify

1. Run `uv run python -m unittest tests.test_main`.
2. Run `uv run ruff check src main.py tests`.
3. Run `uv run pyrefly check`.
4. Run `python3 main.py --list-models --output json` to confirm pure stdlib runtime without `.venv` or PyPI.
