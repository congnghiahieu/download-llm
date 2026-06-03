# download-llm

Download Hugging Face model files, split them into disk-friendly parts, push those parts through Docker Hub, then pull and restore the original files.

## Usage

```bash
uv run python main.py --phases pull_llm,push_docker --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B
```

Common options:

```bash
uv run python main.py \
  --phases pull_llm,push_docker,pull_dockerhub,restore_llm \
  --huggingface-link https://huggingface.co/Qwen/Qwen3.6-27B \
  --docker-namespace hieucien \
  --max-part-size 5GB \
  --max-docker-image-size 5GB
```

Debug keep flags are off by default to save disk: `--keep-raw`, `--keep-parts`, `--keep-images`, `--keep-extracted-parts`.

Docker is called through the Docker CLI. Run `docker login` before `push_docker`.
