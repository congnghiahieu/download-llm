from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from typing_extensions import Self


class HfStdlibTests(unittest.TestCase):
    def test_stdlib_list_model_files_follows_pagination(self) -> None:
        from src import huggingface as hf

        first = [
            {"type": "file", "path": "a.bin"},
            {"type": "folder", "path": "sub"},
            {"type": "file", "path": "sub/b.bin"},
        ]
        second = [{"type": "file", "path": "c.bin"}]

        class FakeResponse:
            def __init__(self, payload: object, link: str | None = None) -> None:
                self._payload = payload
                self.headers = {"Link": link} if link else {}

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        responses = [
            FakeResponse(first, '<https://huggingface.co/api/next>; rel="next"'),
            FakeResponse(second),
        ]
        with (
            patch.object(hf, "_HF_AVAILABLE", False),
            patch("urllib.request.urlopen", side_effect=responses),
        ):
            self.assertEqual(
                hf.list_model_files("Org/Model", "main"),
                ["a.bin", "sub/b.bin", "c.bin"],
            )

    def test_stdlib_download_model_file_writes_bytes(self) -> None:
        from src import huggingface as hf

        class FakeHead:
            headers: ClassVar[dict[str, str]] = {"ETag": '"abc123"', "Content-Length": "6", "Accept-Ranges": "bytes"}

            def read(self) -> bytes:
                return b""

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        class FakeDownload:
            status = 200

            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self, size: int = -1) -> bytes:
                chunk, self._data = self._data[:size], self._data[size:]
                return chunk

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        def fake_urlopen(request: object, timeout: int = 60) -> object:
            if getattr(request, "get_method", lambda: "GET")() == "HEAD":
                return FakeHead()
            return FakeDownload(b"abcdef")

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(cache)}, clear=False),
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
            ):
                dest = hf.download_model_file(
                    "Org/Model", "nested/a.bin", "main", Path(tmp) / "raw"
                )
            self.assertEqual(dest.read_bytes(), b"abcdef")
            blob = cache / "Org_Model" / "main" / "nested" / "a.bin"
            self.assertEqual(blob.read_bytes(), b"abcdef")
            self.assertEqual(
                (blob.parent / "a.bin.etag").read_text(encoding="utf-8").strip(),
                "etag:abc123",
            )

    def test_stdlib_download_model_file_resumes_partial(self) -> None:
        from src import huggingface as hf

        class FakeHead:
            headers: ClassVar[dict[str, str]] = {"ETag": '"abc123"', "Content-Length": "6", "Accept-Ranges": "bytes"}

            def read(self) -> bytes:
                return b""

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        class FakeResume:
            status = 206

            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self, size: int = -1) -> bytes:
                chunk, self._data = self._data[:size], self._data[size:]
                return chunk

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        seen_requests: list[tuple[str, str | None]] = []

        def fake_urlopen(request: object, timeout: int = 60) -> object:
            method = getattr(request, "get_method", lambda: "GET")()
            request_headers = getattr(request, "headers", {})
            seen_requests.append((method, request_headers.get("Range")))
            if method == "HEAD":
                return FakeHead()
            self.assertEqual(request_headers.get("Range"), "bytes=3-")
            return FakeResume(b"def")

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            part_dir = cache / "Org_Model" / "main"
            part_dir.mkdir(parents=True)
            (part_dir / "a.bin.part").write_bytes(b"abc")
            (part_dir / "a.bin.part.etag").write_text("etag:abc123")
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(cache)}, clear=False),
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
            ):
                dest = hf.download_model_file("Org/Model", "a.bin", "main", Path(tmp) / "raw")
            self.assertEqual(dest.read_bytes(), b"abcdef")
            self.assertEqual((part_dir / "a.bin").read_bytes(), b"abcdef")
            self.assertIn(("HEAD", None), seen_requests)

    def test_stdlib_download_model_file_reuses_cache_without_body(self) -> None:
        from src import huggingface as hf

        class FakeHead:
            headers: ClassVar[dict[str, str]] = {"ETag": '"abc123"', "Content-Length": "6", "Accept-Ranges": "bytes"}

            def read(self) -> bytes:
                return b""

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        calls: list[str] = []

        def fake_urlopen(request: object, timeout: int = 60) -> object:
            method = getattr(request, "get_method", lambda: "GET")()
            calls.append(method)
            if method == "HEAD":
                return FakeHead()
            raise AssertionError("body download must not happen on cache hit")

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            blob = cache / "Org_Model" / "main" / "a.bin"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"abcdef")
            (blob.parent / "a.bin.etag").write_text("etag:abc123")
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(cache)}, clear=False),
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
            ):
                dest = hf.download_model_file("Org/Model", "a.bin", "main", Path(tmp) / "raw")
            self.assertEqual(dest.read_bytes(), b"abcdef")
            self.assertEqual(calls, ["HEAD"])

    def test_stdlib_download_model_file_offline_uses_cache(self) -> None:
        from src import huggingface as hf

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            blob = cache / "Org_Model" / "main" / "a.bin"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"abcdef")
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(
                    os.environ,
                    {"DOWNLOAD_LLM_CACHE": str(cache), "HF_HUB_OFFLINE": "1"},
                    clear=False,
                ),
                patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("no network in offline mode"),
                ),
            ):
                dest = hf.download_model_file("Org/Model", "a.bin", "main", Path(tmp) / "raw")
            self.assertEqual(dest.read_bytes(), b"abcdef")


if __name__ == "__main__":
    unittest.main()
