from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote, urlparse

try:
    from huggingface_hub import HfApi as _HfApi
    from huggingface_hub import hf_hub_download as _hf_hub_download

    HfApi: Any = _HfApi
    hf_hub_download: Any = _hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    HfApi = None
    hf_hub_download = None
    _HF_AVAILABLE = False

from src.constants import (
    BUFFER_SIZE,
    DEFAULT_HUGGINGFACE_REVISION,
    HUGGINGFACE_DOMAIN,
    HUGGINGFACE_REPO_TYPE,
)

HF_DEFAULT_ENDPOINT = "https://huggingface.co"
HF_CACHE_ENV = "DOWNLOAD_LLM_CACHE"
HF_OFFLINE_VALUES = ("1", "true", "yes")
_download_locks: dict[tuple[str, str, str], Lock] = {}
_download_locks_guard = Lock()


def _hf_cache_dir() -> Path:
    custom = os.environ.get(HF_CACHE_ENV)
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".cache" / "download-llm" / "hf"


def _safe_cache_segment(value: str) -> str:
    return value.replace("\\", "_").replace("..", "__") or "_"


def _hf_cache_paths(
    repo_id: str, revision: str, filename: str
) -> tuple[Path, Path, Path, Path]:
    root = _hf_cache_dir() / _safe_cache_segment(
        repo_id.replace("/", "_")
    ) / _safe_cache_segment(quote(revision, safe=""))
    rel = Path(
        *[
            _safe_cache_segment(part)
            for part in filename.split("/")
            if part not in ("", ".")
        ]
    )
    blob = root / rel
    return (
        blob,
        blob.with_name(blob.name + ".etag"),
        blob.with_name(blob.name + ".part"),
        blob.with_name(blob.name + ".part.etag"),
    )


def _hf_is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in HF_OFFLINE_VALUES


def _hf_head_meta(url: str, headers: dict[str, str]) -> dict[str, Any] | None:
    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            raw_headers = response.headers or {}
            etag = str(raw_headers.get("ETag") or "").strip().strip('"') or None
            length_raw = raw_headers.get("Content-Length")
            length = (
                int(length_raw)
                if length_raw is not None and str(length_raw).isdigit()
                else None
            )
            accept = str(raw_headers.get("Accept-Ranges") or "").lower()
            return {"etag": etag, "length": length, "ranges": "bytes" in accept}
    except Exception:  # noqa: BLE001 - HEAD is best effort, download proceeds without validation
        return None


def _hf_cache_key(meta: dict[str, Any]) -> str | None:
    if meta.get("etag"):
        return f"etag:{meta['etag']}"
    if meta.get("length") is not None:
        return f"size:{meta['length']}"
    return None


def _hf_cache_valid(etag_file: Path, meta: dict[str, Any]) -> bool:
    key = _hf_cache_key(meta)
    return (
        key is not None
        and etag_file.exists()
        and etag_file.read_text(encoding="utf-8").strip() == key
    )


def _materialize_blob(blob: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(blob, dest)
    except OSError:
        shutil.copyfile(blob, dest)


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


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", HF_DEFAULT_ENDPOINT).rstrip("/") or HF_DEFAULT_ENDPOINT


def _hf_auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token.strip()}"}
    return {}


def _hf_read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:500]
    except Exception:  # noqa: BLE001 - error body is best effort, never fail the report
        return ""


def _hf_open(url: str, headers: dict[str, str] | None = None, timeout: int = 60):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = _hf_read_error_body(exc)
        raise RuntimeError(f"Hugging Face request failed: {url} -> HTTP {exc.code} {detail}".strip())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hugging Face request failed: {url} ({exc.reason})")


def _hf_next_page(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = [segment.strip() for segment in part.split(";")]
        if len(segments) != 2:
            continue
        url_part, rel_part = segments
        if rel_part.strip().lower() != 'rel="next"':
            continue
        if url_part.startswith("<") and url_part.endswith(">"):
            return url_part[1:-1]
    return None


def _list_model_files_stdlib(repo_id: str, revision: str) -> list[str]:
    endpoint = _hf_endpoint()
    url: str | None = (
        f"{endpoint}/api/models/{repo_id}/tree/{quote(revision, safe='')}"
        "?recursive=true&expand=false"
    )
    files: list[str] = []
    while url is not None:
        with _hf_open(url, headers=_hf_auth_headers()) as response:
            payload = json.loads(response.read().decode("utf-8"))
            for item in payload:
                if isinstance(item, dict) and item.get("type") == "file" and item.get("path"):
                    files.append(str(item["path"]))
            url = _hf_next_page(response.headers.get("Link"))
    return files


def model_snapshot(repo_id: str, revision: str) -> tuple[str, list[tuple[str, int]]]:
    """Read the commit and file sizes without downloading file contents."""
    if _hf_is_offline():
        raise RuntimeError("Hugging Face is offline and no pinned snapshot is recorded")
    files: list[tuple[str, int]] = []
    if _HF_AVAILABLE:
        assert HfApi is not None
        api = HfApi()
        commit = api.model_info(repo_id, revision=revision).sha
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("Hugging Face did not return a valid commit SHA")
        for entry in api.list_repo_tree(repo_id, revision=commit, recursive=True):
            size = getattr(entry, "size", None)
            if size is not None:
                files.append((entry.path, size))
    else:
        endpoint = _hf_endpoint()
        headers = _hf_auth_headers()
        info_url = f"{endpoint}/api/models/{repo_id}/revision/{quote(revision, safe='')}"
        with _hf_open(info_url, headers=headers) as response:
            commit = json.load(response).get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("Hugging Face did not return a valid commit SHA")
        url: str | None = f"{endpoint}/api/models/{repo_id}/tree/{commit}?recursive=true"
        while url:
            with _hf_open(url, headers=headers) as response:
                for entry in json.load(response):
                    if entry.get("type") == "file":
                        files.append((entry["path"], entry.get("size")))
                url = _hf_next_page(response.headers.get("Link"))
    if not files or any(not isinstance(size, int) or size < 0 for _, size in files):
        raise RuntimeError("Hugging Face did not return a complete file size inventory")
    return commit, sorted(files)


def planning_cache_dir() -> Path | None:
    # hf_hub_download(local_dir=...) stores the downloaded payload in local_dir.
    return None if _HF_AVAILABLE else _hf_cache_dir()


def cached_download_bytes(repo_id: str, commit: str, filename: str, size: int) -> int:
    if _HF_AVAILABLE:
        return 0
    blob, _, partial, _ = _hf_cache_paths(repo_id, commit, filename)
    if blob.is_file() and blob.stat().st_size == size:
        return size
    if partial.is_file():
        return min(size, partial.stat().st_size)
    return 0


def download_snapshot_file(repo_id: str, filename: str, commit: str, dest: Path) -> Path:
    key = (repo_id, commit, filename)
    with _download_locks_guard:
        lock = _download_locks.setdefault(key, Lock())
    with lock:
        return download_model_file(repo_id, filename, commit, dest)


def _hf_stream_body(
    url: str, filename: str, headers: dict[str, str], target_path: Path, offset: int
) -> None:
    if offset:
        headers = {**headers, "Range": f"bytes={offset}-"}
    try:
        response = _hf_open(url, headers=headers, timeout=60)
    except RuntimeError as exc:
        if offset and "HTTP 416" in str(exc):
            return
        raise
    try:
        status = getattr(response, "status", 200)
        if offset and status == 200:
            mode = "wb"
        elif offset:
            mode = "ab"
        else:
            mode = "wb"
        with response, target_path.open(mode) as target:
            while chunk := response.read(BUFFER_SIZE):
                target.write(chunk)
    except urllib.error.HTTPError as exc:
        detail = _hf_read_error_body(exc)
        raise RuntimeError(f"Hugging Face download failed: {filename} -> HTTP {exc.code} {detail}".strip())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hugging Face download failed: {filename} ({exc.reason})")


def _download_model_file_stdlib(
    repo_id: str, filename: str, revision: str, local_dir: Path
) -> Path:
    endpoint = _hf_endpoint()
    url = f"{endpoint}/{repo_id}/resolve/{quote(revision, safe='')}/{quote(filename)}"
    dest = Path(local_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = _hf_auth_headers()
    try:
        blob, etag_file, part_file, part_etag_file = _hf_cache_paths(
            repo_id, revision, filename
        )
        blob.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError):
        _hf_stream_body(url, filename, headers, dest, dest.stat().st_size if dest.exists() else 0)
        return dest
    if blob.exists():
        if _hf_is_offline():
            _materialize_blob(blob, dest)
            return dest
        meta = _hf_head_meta(url, headers)
        if meta is None:
            offset = part_file.stat().st_size if part_file.exists() else 0
            _hf_stream_body(url, filename, headers, part_file, offset)
            os.replace(part_file, blob)
            if etag_file.exists():
                etag_file.unlink()
            if part_etag_file.exists():
                part_etag_file.unlink()
            _materialize_blob(blob, dest)
            return dest
        if _hf_cache_valid(etag_file, meta):
            _materialize_blob(blob, dest)
            return dest
    else:
        meta = None if _hf_is_offline() else _hf_head_meta(url, headers)
        if meta is None and not _hf_is_offline():
            offset = part_file.stat().st_size if part_file.exists() else 0
            _hf_stream_body(url, filename, headers, part_file, offset)
            os.replace(part_file, blob)
            if etag_file.exists():
                etag_file.unlink()
            if part_etag_file.exists():
                part_etag_file.unlink()
            _materialize_blob(blob, dest)
            return dest
    if _hf_is_offline():
        raise RuntimeError(
            f"Hugging Face offline mode: {filename} is not in the local cache "
            f"({HF_CACHE_ENV} or ~/.cache/download-llm/hf)"
        )
    if meta is None:
        raise RuntimeError(f"Hugging Face request failed: {url} -> no metadata")
    key = _hf_cache_key(meta)
    offset = 0
    if part_file.exists():
        if key is not None and part_etag_file.exists() and part_etag_file.read_text(encoding="utf-8").strip() == key:
            offset = part_file.stat().st_size
        else:
            part_file.unlink()
            if part_etag_file.exists():
                part_etag_file.unlink()
    if key is not None:
        part_etag_file.write_text(key, encoding="utf-8")
    _hf_stream_body(url, filename, headers, part_file, offset)
    if meta.get("length") is not None and part_file.stat().st_size != meta["length"]:
        raise RuntimeError(
            f"Hugging Face download failed: {filename} is truncated "
            f"({part_file.stat().st_size}B of {meta['length']}B)"
        )
    os.replace(part_file, blob)
    if key is not None:
        etag_file.write_text(key, encoding="utf-8")
    if part_etag_file.exists():
        part_etag_file.unlink()
    _materialize_blob(blob, dest)
    return dest


def list_model_files(repo_id: str, revision: str) -> list[str]:
    if _HF_AVAILABLE:
        assert HfApi is not None
        api = HfApi()
        return api.list_repo_files(
            repo_id=repo_id, revision=revision, repo_type=HUGGINGFACE_REPO_TYPE
        )
    return _list_model_files_stdlib(repo_id, revision)


def download_model_file(
    repo_id: str, filename: str, revision: str, local_dir: Path
) -> Path:
    if _HF_AVAILABLE:
        assert hf_hub_download is not None
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=HUGGINGFACE_REPO_TYPE,
                local_dir=local_dir,
            )
        )
    return _download_model_file_stdlib(repo_id, filename, revision, local_dir)
