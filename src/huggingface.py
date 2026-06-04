from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download

from src.constants import (
    DEFAULT_HUGGINGFACE_REVISION,
    HUGGINGFACE_DOMAIN,
    HUGGINGFACE_REPO_TYPE,
)


def parse_huggingface_url(url: str, revision: str | None) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.netloc != HUGGINGFACE_DOMAIN:
        raise ValueError("Hugging Face URL must use huggingface.co")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face URL must include owner and repo name")
    repo_id = "/".join(parts[:2])
    url_revision = None
    if len(parts) >= 4 and parts[2] == "tree":
        url_revision = "/".join(parts[3:])
    return repo_id, revision or url_revision or DEFAULT_HUGGINGFACE_REVISION


def list_model_files(repo_id: str, revision: str) -> list[str]:
    api = HfApi()
    return api.list_repo_files(
        repo_id=repo_id, revision=revision, repo_type=HUGGINGFACE_REPO_TYPE
    )


def download_model_file(
    repo_id: str, filename: str, revision: str, local_dir: Path
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=HUGGINGFACE_REPO_TYPE,
            local_dir=local_dir,
        )
    )
