# download-llm

Download Hugging Face model files, split them into disk-friendly parts, push those parts through Docker Hub, then pull and restore the original files.

## Modes

1. Use non-interactive mode with explicit flags for scripts and repeatable commands.
2. Use interactive mode with no arguments for guided prompts with safe defaults.
3. Use `uv run python main.py` for dev and `python3 main.py` for pure stdlib runtime without `.venv` or PyPI.

## Common usage

### Transfer restored models over SSH

Use the standalone transfer workflow, or select workflow 5 in interactive mode.

```bash
python3 main.py --transfer-models \
  --model-name Toto-2.0-1B,Toto-2.0-2.5B \
  --ssh-host destination-alias \
  --destination-path /home/vht/model_weights
```

1. Copies the contents of `<weights-dir>/<model>/extracted/restored/` directly into `<destination-path>/<model>/`, including hidden files and empty directories, without source state, parts or Docker metadata.
2. Uses existing model aliases and `--model-weights-dir` / `MODEL_WEIGHTS_DIR`; every selected model must have restore status `done` and valid manifest file sizes before any transfer starts.
3. Defaults to parallel execution, with one worker per model; use `--max-workers N` to limit concurrency or `--max-workers 1` for sequential execution in input order.
4. Requires OpenSSH locally and an SSH server remotely; prefers rsync on both ends, falling back to SCP only when rsync is missing on either end.
5. SCP fallback requires `scp` on both ends, uses the legacy protocol for compatibility with Ubuntu 22.04 and newer clients, and copies files again without rsync-style resume or unchanged-file skipping.
6. OpenSSH handles config aliases, agent, keys, password, MFA and host-key verification through the terminal; the script reuses a private master connection and closes it on completion.
7. Repeat `--ssh-option Port=2222`, `--ssh-option IdentityFile=/path/to/key` or `--ssh-option ProxyJump=bastion` for additional SSH settings; connection multiplexing and session-control options are managed by the script.
8. Destination paths must be absolute (no `~`); spaces and shell metacharacters are quoted safely, and absent model directories are created during real transfers.
9. Checks each destination model folder before starting workers; if non-empty, offers a new destination root (recommended), merge, or cancel, and rechecks a new root for the entire batch.
10. `--existing-destination ask` is the default; without a terminal a conflict fails safely, unless `--existing-destination merge` explicitly permits overwriting matching files while retaining extras.
11. `--dry-run` makes no remote payload writes: rsync previews differences for existing folders; new folders are reported as an all-files copy, and SCP only reports the transfer plan, not a diff.
12. Preserves permissions and timestamps where allowed, but not owner/group; symlinks in restored payloads and symlink model destination folders are rejected to avoid backend-dependent behavior.
13. Reports per-model results and returns nonzero if any model fails, while continuing other models; transfer does not modify `state.json` or add a fifth phase.
14. Do not modify source or destination folders concurrently with a transfer; preflight checks do not lock remote directories, and interrupted/failed transfers may leave partial data.

### Download and restore

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

### Preview and schedule phase execution

Use `--plan` to preview a run without downloading weights, writing model state, or building and pushing images.
It fetches Hugging Face metadata for new downloads and checks the relevant storage and local Docker image inventory.

```bash
python3 main.py --plan --output json --phases pull_llm,push_docker \
  --huggingface-link https://huggingface.co/owner/model --max-workers 2
python3 main.py --plan --phases pull_dockerhub,restore_llm --model-name my-model
```

1. `--max-workers` is the maximum number of active models, not a multiplier for task limits.
2. The initial command-wide limits are 2 Hugging Face downloads, 1 split, 1 build, 1 push, 2 Docker pulls, and 1 shared extract/restore slot.
3. No-args interactive mode shows the plan before execution and lets you revise the model and task limits for that run.
4. The planner distinguishes final storage from temporary peak per filesystem and reserves 1 GiB of headroom per filesystem.
5. A plan that does not fit at the requested concurrency can reduce model concurrency or serialize tasks; known insufficient storage blocks execution.
6. If local Docker storage needs elevated read access, execution may ask permission to use `sudo stat`; the application does not handle the password.
7. Unknown storage requires explicit confirmation before execution; without a terminal the command exits nonzero instead.
8. Docker layer/cache and network estimates are approximate; runtime admission rechecks free space and stops with checkpoints if no task can safely proceed.

In `pull_llm,push_docker`, file downloads and splits can overlap with image builds and pushes as soon as each group has its parts.
Builds use an isolated context; newly published pinned models record Docker image digests in `state.json` and are ready for the receiving machine only after the complete manifest is saved.
For `pull_dockerhub,restore_llm`, a pulled image can be extracted immediately, and a file is restored when all its parts are present and verified.
Pass the sender's completed `state.json` to the receiving machine as before; `pull_dockerhub` alone still only pulls images.
For a legacy download already in progress without a pinned commit, use a new model directory for `pull_llm`; existing completed legacy states can still be pushed or restored as appropriate.

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

Execution logs record run and per-model task status and timings on stderr and in `logs/download-llm.log` relative to the current directory.
If the log file cannot be opened, execution continues with console logging only.
Progress bars, Docker output, and list-models JSON are not copied into the log file.

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
