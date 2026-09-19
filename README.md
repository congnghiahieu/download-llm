# download-llm

Download Hugging Face model files, split them into disk-friendly parts, push those parts through Docker Hub, then pull and restore the original files.

## Modes

1. Use non-interactive mode with explicit flags for scripts and repeatable commands.
2. Use interactive mode with no arguments for guided prompts with safe defaults.
3. Use `uv run python main.py` for dev and `python3 main.py` for pure stdlib runtime without `.venv` or PyPI.

## Common usage

Run interactive mode with no arguments and follow the prompts.

```bash
# Start interactive mode, pick flow 1 or flow 2, then confirm to execute.
uv run python main.py
```

Export `HF_TOKEN` before flow 1 for faster `pull_llm`, especially for gated models.

```bash
# Single model download and ship, with token for speed.
export HF_TOKEN=hf_xxx
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B
```

```bash
# Two more flow 1 examples, same pattern with different links.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-35B-A3B
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/google/gemma-4-31B-it
```

Pull and restore a single model with flow 2.

```bash
# Restore by model name, resolved against model_weights/*/state.json.
uv run python main.py --phases pull_dockerhub,restore_llm --model-name Qwen3.6-35B-A3B
```

Pull via a proxy registry instead of direct Docker Hub access.

```bash
# --docker-pull-prefix adds a registry prefix, or set DOCKER_PULL_PREFIX env var.
uv run python main.py --phases pull_dockerhub,restore_llm --model-name Qwen3.6-35B-A3B --docker-pull-prefix docker.internal/proxy-cache
DOCKER_PULL_PREFIX=docker.internal/proxy-cache uv run python main.py --phases pull_dockerhub,restore_llm --model-name Qwen3.6-35B-A3B
```

Run `docker login` before `push_docker` because Docker is called through the Docker CLI.

## List models

```bash
# Rich table with model name, 4-phase done or pending status, raw file count, raw size est, image count, and image size.
uv run python main.py --list-models
```

```bash
# JSON output for scripts, same fields plus byte counts and unknown image count.
uv run python main.py --list-models --output json
```

```bash
# Custom weights folder via flag, or set MODEL_WEIGHTS_DIR env var, applies to every phase and to list.
uv run python main.py --list-models --model-weights-dir /data/weights
MODEL_WEIGHTS_DIR=/data/weights uv run python main.py --list-models
```

Image size is recorded into `state.json` when each image is built.

Models built before this feature show `unknown` for image size.

## Multiple models

```bash
# Two links via repeated flags, parallel by default. One --revision applies to every link.
# An explicit --model-name is honored for a single link and ignored with a warning for multiple links.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B --huggingface-link https://huggingface.co/google/gemma-4-31B-it
```

```bash
# Same two links comma-separated, capped at 2 parallel workers.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B,https://huggingface.co/google/gemma-4-31B-it --max-workers 2
```

Use `--max-workers 1` to run multiple targets sequentially in input order.

```bash
# Sequential pull and ship, one pipeline at a time.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B,https://huggingface.co/google/gemma-4-31B-it --max-workers 1
```

```bash
# Two models in parallel. Aliases accept model_name, folder, docker_repo_name, repo_id, or url.
uv run python main.py --phases pull_dockerhub,restore_llm --model-name Qwen3.6-27B,Qwen3.8-27B --max-workers 2
```

```bash
# Same two models sequentially via repeated flags.
uv run python main.py --phases pull_dockerhub,restore_llm --model-name Qwen3.6-27B --model-name Qwen3.8-27B --max-workers 1
```

## Failures and retry

Config and precondition errors fail fast before any work starts.

```bash
# Invalid phase, bad link, unknown model, or missing docker daemon all stop immediately.
uv run python main.py --phases pull_llm,push_docker --huggingface-link not-a-link
```

Runtime errors during multi-model runs are best effort in both modes.

```bash
# If link A fails with a network error, link B still runs, then the report lists A with its retry command.
uv run python main.py --phases pull_llm --huggingface-link https://huggingface.co/Qwen/A,https://huggingface.co/Org/B --max-workers 1
```

Rerun the printed per-model retry command to redo only the failed model. The prefix adapts to the runtime: `uv run python main.py` inside `.venv` or `uv` projects, else `python3 main.py`.

```bash
# Example retry output for a failed model, rerun exactly this to resume it.
uv run python main.py --phases pull_llm --huggingface-link https://huggingface.co/Qwen/A --docker-namespace hieucien --max-part-size 5GB --max-docker-image-size 5GB
```

Rerunning the original multi-model command is also safe because completed files and entries are skipped via `state.json`.

## State and resume

Progress is tracked per model in `model_weights/<model>/state.json` and saved after every file and entry.

`pull_llm` skips files that are already downloaded and split.

`push_docker` skips images that are already built and pushed.

`pull_dockerhub` skips images that are already pulled and present locally.

`restore_llm` skips parts that are already extracted and outputs that are already valid.

## Interactive mode

Interactive mode asks for weights dir, workflow, token, links or models, sizes, keep flags, prefix, execution mode, then confirmation.

```bash
# Workflow 1 downloads and ships, workflow 2 pulls and restores, custom allows single phases, list shows the model table.
uv run python main.py
```

Interactive mode prompts for size and keep flags with safe defaults instead of hardcoding them.

```bash
# Defaults prompted in interactive mode, pass them explicitly in CLI to override.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B --max-part-size 5GB --max-docker-image-size 5GB
```

Debug keep flags are off by default to save disk.

```bash
# Keep flags for debugging, available in both non-interactive and interactive modes.
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B --keep-raw --keep-parts --keep-images --keep-extracted-parts
```

## Full pipeline example

Run every phase for one model with explicit namespace and sizes.

```bash
# All four phases in one command, useful for end-to-end verification.
uv run python main.py --phases pull_llm,push_docker,pull_dockerhub,restore_llm --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B --docker-namespace hieucien --max-part-size 5GB --max-docker-image-size 5GB
```
