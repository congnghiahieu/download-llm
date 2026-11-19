from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.cleanup import apply_cleanup, make_cleanup_plan
from src.cli import execute, parse_args
from src.console import Console
from src.huggingface import _hf_cache_paths, snapshot_cache_lock
from src.interactive import prompt_cleanup
from src.state import load_state, make_initial_state, model_lock, save_state


class CleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(root)
        self.model = self.root / "model"
        self.model.mkdir()
        self.repo = "org/model"
        self.commit = "a" * 40
        self.state = make_initial_state(
            "model", self.repo, "main", "https://huggingface.co/org/model", 2, 2
        )
        self.state["snapshot"] = {"commit": self.commit, "files": [["file.bin", 4]]}
        self.state["raw"] = [
            {"filename": "file.bin", "downloaded": True, "split": True}
        ]
        self.state["parts"] = [
            {"part_filename": "file-part0001.bin", "extracted": True}
        ]
        self.state["restore"] = [{"raw_filename": "file.bin", "restored": True}]
        save_state(self.model, self.state)
        self.logs, self.errors = io.StringIO(), io.StringIO()
        stack.enter_context(redirect_stdout(self.logs))
        stack.enter_context(redirect_stderr(self.errors))
        stack.enter_context(
            patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(self.root / "cache")})
        )

    def args(self, *extra: str):
        return parse_args(
            [
                "--cleanup",
                "--model-name",
                "model",
                "--model-weights-dir",
                str(self.root),
                *extra,
            ]
        )

    def write(self, name: str, data: bytes = b"1234") -> Path:
        path = self.model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_cli_requires_exact_selection_and_yes(self) -> None:
        for flags in (
            ("--delete", "raw"),
            ("--yes",),
            ("--delete", "state", "--yes"),
            ("--delete", "dockerfiles", "--yes"),
            ("--phases", "pull_llm", "--dry-run"),
            ("--model-name", "other", "--dry-run"),
        ):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.args(*flags)

    def test_dry_run_measures_and_does_not_write(self) -> None:
        raw = self.write("raw/file.bin")
        state = (self.model / "state.json").read_bytes()
        self.assertEqual(execute(self.args("--delete", "raw", "--dry-run")), 0)
        self.assertIn("raw: 4B", self.logs.getvalue())
        self.assertIn(str(raw.parent), self.logs.getvalue())
        self.assertEqual(raw.read_bytes(), b"1234")
        self.assertEqual((self.model / "state.json").read_bytes(), state)
        self.assertFalse((self.model / ".pipeline.lock").exists())

    def test_artifact_selections_keep_state_dockerfiles_and_restore_independent(
        self,
    ) -> None:
        raw = self.write("raw/file.bin")
        part = self.write("parts/file-part0001.bin")
        extracted = self.write("extracted/parts/file-part0001.bin")
        restored = self.write("extracted/restored/file.bin")
        dockerfile = self.write("dockerfiles/Dockerfile.part0001")
        self.assertEqual(execute(self.args("--delete", "parts,extracted", "--yes")), 0)
        self.assertTrue(all(path.is_file() for path in (raw, restored, dockerfile)))
        self.assertFalse(part.exists())
        self.assertFalse(extracted.exists())
        state = load_state(self.model)
        self.assertTrue(state["parts"][0]["deleted"])
        self.assertFalse(state["parts"][0]["extracted"])
        self.assertTrue(state["restore"][0]["restored"])
        self.assertEqual(state["snapshot"]["commit"], self.commit)

    def test_active_pipeline_and_preview_changes_prevent_deletion(self) -> None:
        raw = self.write("raw/file.bin")
        args = self.args("--delete", "raw", "--yes")
        with model_lock(self.model):
            self.assertEqual(execute(args), 1)
            self.assertIn("owns this model directory", self.errors.getvalue())
        preview = make_cleanup_plan(args)
        raw.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed since preview"):
            apply_cleanup(preview, {"raw"})
        self.assertEqual(raw.read_bytes(), b"changed")

    def test_cache_scope_includes_partial_and_local_hub_cache(self) -> None:
        raw = self.write("raw/file.bin")
        local = self.write("raw/.cache/huggingface/download/file.incomplete")
        blob, etag, partial, partial_etag = _hf_cache_paths(
            self.repo, self.commit, "file.bin"
        )
        cache_files = (blob, etag, partial, partial_etag)
        other, *_ = _hf_cache_paths(self.repo, "b" * 40, "file.bin")
        extra = blob.with_name("untracked.bin")
        for path in (*cache_files, other, extra):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"12")
        args = self.args("--delete", "hf-cache", "--yes")
        with snapshot_cache_lock(self.repo, self.commit, wait=True):
            self.assertEqual(execute(args), 1)
            self.assertTrue(partial.exists())
            self.assertTrue(local.exists())
        self.assertEqual(execute(args), 0)
        self.assertFalse(any(path.exists() for path in (*cache_files, local)))
        self.assertTrue(all(path.is_file() for path in (raw, other, extra)))

    def test_symlinked_directory_is_not_followed(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        payload = outside / "file.bin"
        payload.write_bytes(b"preserve")
        (self.model / "raw").symlink_to(outside, target_is_directory=True)
        self.assertEqual(execute(self.args("--delete", "raw", "--yes")), 1)
        self.assertTrue(payload.is_file())

    def test_partial_failure_keeps_state_aligned_with_each_path(self) -> None:
        import shutil

        raw = self.write("raw/file.bin")
        part = self.write("parts/file-part0001.bin")
        rmtree = shutil.rmtree

        def remove(path: Path) -> None:
            if path.name == "parts":
                raise PermissionError("parts are read-only")
            rmtree(path)

        with patch("src.cleanup.shutil.rmtree", side_effect=remove):
            self.assertEqual(execute(self.args("--delete", "raw,parts", "--yes")), 1)
        state = load_state(self.model)
        self.assertFalse(raw.exists())
        self.assertTrue(state["raw"][0]["deleted"])
        self.assertTrue(part.exists())
        self.assertFalse(state["parts"][0]["deleted"])
        self.assertIn("parts are read-only", self.errors.getvalue())

    def test_image_cleanup_uses_only_confirmed_refs_and_keeps_publication(self) -> None:
        digest = "sha256:" + "c" * 64
        tag = "org/model:part0001"
        self.state["dockerfiles"] = [
            {"tag": tag, "digest": digest, "built": True, "pushed": True}
        ]
        self.state["publication"] = {"complete": True}
        save_state(self.model, self.state)
        removed = []

        def docker(*command: str) -> subprocess.CompletedProcess[str]:
            if command[0] == "context":
                return subprocess.CompletedProcess(
                    command, 0, "unix:///run/docker.sock", ""
                )
            if command[:2] == ("image", "inspect"):
                return (
                    subprocess.CompletedProcess(command, 0, "sha256:local 500", "")
                    if command[-1] == tag
                    else subprocess.CompletedProcess(
                        command, 1, "", "Error: No such image: absent"
                    )
                )
            self.assertEqual(command, ("image", "rm", "--no-prune", "--", tag))
            removed.append(command[-1])
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch.dict(os.environ, {"DOCKER_HOST": "", "DOCKER_CONTEXT": ""}),
            patch("src.cleanup._docker", side_effect=docker),
        ):
            self.assertEqual(execute(self.args("--delete", "images", "--yes")), 0)
        self.assertEqual(removed, [tag])
        result = load_state(self.model)
        self.assertTrue(result["dockerfiles"][0]["removed"])
        self.assertTrue(result["dockerfiles"][0]["pushed"])
        self.assertEqual(result["dockerfiles"][0]["digest"], digest)
        self.assertTrue(result["publication"]["complete"])

    def test_image_failure_is_reported_without_marking_image_removed(self) -> None:
        self.state["dockerfiles"] = [{"tag": "org/model:part0001", "built": True}]
        save_state(self.model, self.state)

        def docker(*command: str) -> subprocess.CompletedProcess[str]:
            if command[1] == "inspect":
                return subprocess.CompletedProcess(command, 0, "sha256:local 500", "")
            return subprocess.CompletedProcess(
                command, 1, "", "image is being used by a container"
            )

        with (
            patch("src.cleanup._docker_local"),
            patch("src.cleanup._docker", side_effect=docker),
        ):
            self.assertEqual(execute(self.args("--delete", "images", "--yes")), 1)
        self.assertIn("being used by a container", self.errors.getvalue())
        self.assertFalse(load_state(self.model)["dockerfiles"][0]["removed"])

    def test_interactive_selection_has_no_default(self) -> None:
        raw = self.write("raw/file.bin")
        part = self.write("parts/file-part0001.bin")
        with (
            patch("src.interactive.Prompt.ask", side_effect=["model", ""]),
            patch("src.interactive.Confirm.ask") as confirm,
        ):
            self.assertEqual(prompt_cleanup(Console(), str(self.root)), 0)
        confirm.assert_not_called()
        self.assertTrue(raw.exists())
        with (
            patch("src.interactive.Prompt.ask", side_effect=["model", "1", ""]),
            patch("src.interactive.Confirm.ask", return_value=True),
        ):
            self.assertEqual(prompt_cleanup(Console(), str(self.root)), 0)
        self.assertFalse(raw.exists())
        self.assertTrue(part.exists())
