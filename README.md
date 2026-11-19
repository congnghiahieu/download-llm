# download-llm

Download Hugging Face model files, split them into disk-friendly parts, push those parts through Docker Hub, then pull and restore the original files.

## Modes

1. Use non-interactive mode with explicit flags for scripts and repeatable commands.
2. Use interactive mode with no arguments for guided prompts with safe defaults.
3. Use `uv run python main.py` for dev and `python3 main.py` for pure stdlib runtime without `.venv` or PyPI.

## Common usage

### Clean up local model data

Select workflow 6 in interactive mode to inspect each available category, toggle the ones to remove, and confirm the selection. For the CLI, preview first, then explicitly pass `--delete` and `--yes` to delete:

```bash
python3 main.py --cleanup --model-name MyModel --dry-run
python3 main.py --cleanup --model-name MyModel --delete raw,parts,hf-cache --yes
```

1. Cleanup targets one model and supports `--model-weights-dir` / `MODEL_WEIGHTS_DIR`. Categories are `raw`, `parts`, `images`, `extracted`, `restored`, and `hf-cache`; `extracted` removes only `extracted/parts`, not `extracted/restored`.
2. Preview inspects actual paths and local Docker references and reports their sizes. Shared Docker layers and hard-linked files mean the reported sizes are not guaranteed disk savings. Image cleanup requires a local Docker socket and never forces deletion, prunes Docker data, or contacts the registry.
3. `hf-cache` includes the model's `raw/.cache/huggingface` directory and exact files for its pinned snapshot under `DOWNLOAD_LLM_CACHE` (default `~/.cache/download-llm/hf`), including unfinished downloads. Removing `raw` also removes the nested cache. Global Hugging Face Hub and Xet caches are not removed.
4. Cleanup refuses to delete while the model's pipeline is active, or while another download uses the shared snapshot cache being deleted. It retains `state.json`, Dockerfiles, and publication metadata; statuses such as `push_docker: done` do not imply that local images are still present.
5. To recover local data, rerun the relevant phase with the original settings. Use `pull_llm --keep-raw` for raw files, `pull_llm,push_docker` if unpublished images need missing parts, or `restore_llm` for restored files. The pipeline reuses valid artifacts, uses the pinned commit or Docker digest when it needs to fetch, and skips images already published. Legacy states without the pins required to reproduce deleted data do not silently fall back to a movable reference.

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

## Restore and deliver to another host

```bash
python3 main.py --phases pull_dockerhub,restore_llm,deliver_llm \
  --model-name Qwen3.6-35B-A3B --ssh-host model-server --destination-path /srv/models

# Resume delivery from local restored files and the remote staging directory; no Docker calls.
python3 main.py --phases deliver_llm \
  --model-name Qwen3.6-35B-A3B --ssh-host model-server --destination-path /srv/models
```

1. Delivery requires OpenSSH and `rsync` on the intermediate host, and `rsync` and `python3` on the Linux destination; no application install is needed there. Configure non-interactive SSH authentication and host verification up front (for example, via `ssh-agent` and `known_hosts`).
2. One independent worker sends the largest ready file across all selected models. Restore always stages complete verified files locally and does not slow down to match SSH. Temporary network failures get up to two retries per command; authentication and destination conflicts are not retried.
3. Files go into `/srv/models/<model>.transferring`. The receiver retains partial files for `rsync` to reuse, checks SHA-256, and publishes the directory to `/srv/models/<model>` only after the full manifest matches. A matching final directory is accepted; a conflicting final directory or staging manifest is never overwritten.
4. Each file is removed from the intermediate host only after its remote copy is verified, unless `--keep-local` is set. If transfer fails, unverified files stay local, and restore continues as disk allows. If a previously verified remote file is later lost after local cleanup, run `restore_llm` separately to recreate it before resuming delivery.
5. The plan shows the disk estimate without transfer cleanup. During the run, the planner reports staged backlog and estimates additional peak per filesystem both with continued transfer and with no further cleanup. The rolling estimate becomes available after throughput samples and includes the planner's conservative Docker/extraction working-space allowance. Neither backlog nor forecast throttles restore; actual lack of space stops new restore work and reports the remaining files and disk shortfall. Delivery can still run for already staged files.
6. Restore and delivery report separate outcomes, and a combined run exits nonzero if either is incomplete. `--transfer-models` remains the separate workflow for copying an already restored model tree. The no-args menu also offers the combined pipeline and delivery-only resume.

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
2. Command-wide task caps are calculated for each run from pending jobs, measured CPU and available RAM, and measured free space. Extract and restore share one operation limit.
3. No-args interactive mode shows the plan before execution and lets you revise the model and task limits for that run.
4. The planner distinguishes final storage from temporary peak per filesystem and reserves 1 GiB of headroom per filesystem.
5. A plan that does not fit at the requested concurrency can reduce model concurrency or serialize tasks; known insufficient storage blocks execution except for delivery pipelines, which check each task at runtime and can reclaim verified local files.
6. If local Docker storage needs elevated read access, execution may ask permission to use `sudo stat`; the application does not handle the password.
7. Unknown storage requires explicit confirmation before execution; without a terminal the command exits nonzero instead.
8. Docker layer/cache and network estimates are approximate; runtime admission rechecks free space and stops with checkpoints if no task can safely proceed.
9. If the scheduler cannot admit new work because a disk probe fails, it waits for in-flight tasks and checkpoints successful results before reporting the failure. A forced process termination cannot guarantee this.
10. If the Docker daemon reports a containerd image store, push streams existing layers and does not reserve another image-sized allocation or probe disk at task admission. If the backend cannot be verified or uses the classic store, push keeps the conservative reservation and probe. Preflight and admission checks for write-heavy tasks such as build, pull, and restore are unchanged.

The table starts with the disk decision and shows `Extra kept` (additional bytes still used after the run) and `Extra at peak` (the largest additional requirement during the run).
The per-model section also reports the original model size, which is not necessarily the amount kept locally. Use the combined table to judge a multi-model run; individual peaks are estimates for a model running on its own.
The operation table shows the remaining jobs and shared caps for the selected phases. CPU and RAM costs per job are estimates, not measured job usage or a throughput benchmark.
The planner starts with one slot per operation and only adds slots if enough jobs remain and the disk and compute budgets permit it. Unknown disk capacity prevents scaling up and still requires confirmation before execution.
JSON output also includes network estimates, capacity inputs, and both the configured and effective task limits.

On WSL 2, the planner checks the Linux filesystem and the Windows drive backing its VHDX separately.
For Docker Desktop with its WSL backend, it measures the daemon's guest filesystem and checks the Windows drive containing `docker_data.vhdx`.
Paths already on a Windows-mounted drive are counted only on that drive. If Ubuntu, the Hugging Face cache, and Docker use the same Windows drive, their `*_host` roles share one budget and one headroom.
Host estimates conservatively reserve space for task writes; deleting guest files does not imply that Windows gets the space back. Free bytes are rechecked before admitting each task.
The probe uses read-only Windows interop with the WSL registry and Docker Desktop settings; unsupported layouts, disabled interop, or unreadable mounts are shown as `unknown` rather than substituted with the Linux root's free space.

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
The file also records the approved plan, task admissions, Docker stdout/stderr and exit codes, failure tracebacks, storage measurements at failure, and retry commands.
Use the `run` and `command` IDs to trace concurrent failures. Known Hugging Face tokens, authorization values and HTTP URL credentials/query strings are redacted in the log.
If Docker storage is `unknown`, the weights filesystem's free bytes do not establish how much disk space the Docker daemon had.
Interactive progress bars and list-models JSON are not copied into the log file.

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

`pull_llm` skips files whose requested outputs already exist and are valid; it regenerates missing outputs when rerun after cleanup.

`push_docker` skips images that are already built and pushed.

`pull_dockerhub` skips images that are already pulled and present locally.

`restore_llm` reuses valid extracted parts and restored files, even if writing their checkpoints was interrupted.

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
