from __future__ import annotations

import hashlib
import io
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src.phases import PHASE_HANDLERS
from scripts import src_image


class DownloadLlmTests(unittest.TestCase):
    def test_parse_huggingface_root_url_defaults_main(self) -> None:
        repo_id, revision = main.parse_huggingface_url(
            "https://huggingface.co/Qwen/Qwen3.6-27B", None
        )
        self.assertEqual(repo_id, "Qwen/Qwen3.6-27B")
        self.assertEqual(revision, "main")

    def test_parse_huggingface_tree_url_uses_tree_revision(self) -> None:
        repo_id, revision = main.parse_huggingface_url(
            "https://huggingface.co/Qwen/Qwen3.6-27B/tree/dev", None
        )
        self.assertEqual(repo_id, "Qwen/Qwen3.6-27B")
        self.assertEqual(revision, "dev")

    def test_parse_size_units(self) -> None:
        self.assertEqual(main.parse_size("5GB"), 5_000_000_000)
        self.assertEqual(main.parse_size("2MiB"), 2 * 1024 * 1024)

    def test_main_reexports_parse_size_and_internal_phase_handlers(self) -> None:
        self.assertEqual(main.parse_size("1KB"), 1000)
        self.assertIn("pull_llm", PHASE_HANDLERS)

    def test_split_and_restore_streaming(self) -> None:
        payload = b"abcdefghi"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "nested" / "model.bin"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(payload)

            raw_sha, parts = main.split_raw_file(
                raw, Path("nested/model.bin"), root / "parts", 4
            )
            self.assertEqual(raw_sha, hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                [part["part_filename"] for part in parts],
                [
                    "nested/model-part0001.bin",
                    "nested/model-part0002.bin",
                    "nested/model-part0003.bin",
                ],
            )

            raw_meta = {
                "filename": "nested/model.bin",
                "size_bytes": len(payload),
                "sha256": raw_sha,
            }
            extracted = root / "model" / "extracted" / "parts" / "nested"
            extracted.mkdir(parents=True)
            for part in parts:
                source = root / "parts" / part["part_filename"]
                target = root / "model" / "extracted" / "parts" / part["part_filename"]
                target.write_bytes(source.read_bytes())

            restored = main.restore_raw_file(root / "model", raw_meta, parts)
            self.assertTrue(restored["restored"])
            self.assertEqual(
                (
                    root / "model" / "extracted" / "restored" / "nested" / "model.bin"
                ).read_bytes(),
                payload,
            )

    def test_group_parts_and_dockerfile_labels(self) -> None:
        parts = [
            {"part_filename": "a-part0001.bin", "size_bytes": 5},
            {"part_filename": "a-part0002.bin", "size_bytes": 5},
            {"part_filename": "a-part0003.bin", "size_bytes": 2},
        ]
        groups = main.group_parts(parts, 10)
        self.assertEqual(
            [[part["part_filename"] for part in group] for group in groups],
            [
                ["a-part0001.bin", "a-part0002.bin"],
                ["a-part0003.bin"],
            ],
        )
        self.assertEqual(main.part_label(7), "part0007")

    def test_confirm_unpushed_yes_all_and_no_all(self) -> None:
        with (
            patch("builtins.input", return_value="ya"),
            patch("sys.stderr", io.StringIO()),
        ):
            should_pull, decision = main.confirm_unpushed("repo/model:part0001", None)
        self.assertTrue(should_pull)
        self.assertEqual(decision, "yes_all")
        should_pull, decision = main.confirm_unpushed("repo/model:part0002", "no_all")
        self.assertFalse(should_pull)
        self.assertEqual(decision, "no_all")

    def test_join_docker_prefix(self) -> None:
        self.assertEqual(
            main.join_docker_prefix(
                "docker.internal/proxy-cache", "hieucien/model:part0001"
            ),
            "docker.internal/proxy-cache/hieucien/model:part0001",
        )
        self.assertEqual(
            main.join_docker_prefix(
                "docker.internal/proxy-cache/",
                "docker.internal/proxy-cache/hieucien/model:part0001",
            ),
            "docker.internal/proxy-cache/hieucien/model:part0001",
        )

    def test_docker_pull_tag_reuses_recorded_proxy_tag(self) -> None:
        entry = {
            "tag": "hieucien/model:part0001",
            "pull_tag": "docker.internal/proxy-cache/hieucien/model:part0001",
        }
        self.assertEqual(
            main.docker_pull_tag(entry),
            "docker.internal/proxy-cache/hieucien/model:part0001",
        )
        self.assertEqual(
            main.docker_pull_tag(entry, "docker.internal/proxy-cache"),
            "docker.internal/proxy-cache/hieucien/model:part0001",
        )

    def test_src_image_tag_with_proxy_prefix(self) -> None:
        self.assertEqual(
            src_image.parse_image_tag(
                "docker.internal/proxy-cache/hieucien/download-llm-src:2026-06-03_17-22-20"
            ),
            "2026-06-03_17-22-20",
        )

    def test_pull_llm_resumes_downloaded_raw_and_skips_split_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "ResumeModel"
            raw_dir = model_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "a.bin").write_bytes(b"abcdef")
            state = main.make_initial_state(
                "ResumeModel", "Org/ResumeModel", "main", "url", 4, 10
            )
            state["raw"] = [
                {
                    "filename": "a.bin",
                    "path": (raw_dir / "a.bin").as_posix(),
                    "downloaded": True,
                    "split": False,
                    "deleted": False,
                },
                {
                    "filename": "b.bin",
                    "path": (raw_dir / "b.bin").as_posix(),
                    "downloaded": True,
                    "split": True,
                    "deleted": True,
                    "size_bytes": 1,
                    "size": "1B",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                },
            ]
            state["parts"] = [
                {
                    "raw_filename": "b.bin",
                    "index": 1,
                    "part_filename": "b-part0001.bin",
                    "path": (model_dir / "parts" / "b-part0001.bin").as_posix(),
                    "size_bytes": 1,
                    "size": "1B",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                    "deleted": False,
                    "extracted": False,
                    "restored": False,
                }
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                huggingface_link="https://huggingface.co/Org/ResumeModel",
                revision=None,
                model_name="ResumeModel",
                max_part_size_bytes=4,
                max_docker_image_size_bytes=10,
                keep_raw=False,
            )

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.list_model_files", return_value=["a.bin", "b.bin"]),
                patch("src.phases.download_model_file") as download,
            ):
                main.phase_pull_llm(args)

            download.assert_not_called()
            saved = main.load_state(model_dir)
            raw_by_name = {item["filename"]: item for item in saved["raw"]}
            self.assertTrue(raw_by_name["a.bin"]["split"])
            self.assertTrue(raw_by_name["a.bin"]["deleted"])
            self.assertTrue(raw_by_name["b.bin"]["split"])
            self.assertEqual(
                [part["raw_filename"] for part in saved["parts"]].count("b.bin"), 1
            )

    def test_push_docker_rebuilds_missing_built_image_and_marks_missing_part_deleted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "PushModel"
            state = main.make_initial_state("PushModel", "Org/PushModel", "main", "url", 4, 10)
            state["parts"] = [
                {
                    "raw_filename": "a.bin",
                    "index": 1,
                    "part_filename": "a-part0001.bin",
                    "path": (model_dir / "parts" / "a-part0001.bin").as_posix(),
                    "size_bytes": 1,
                    "size": "1B",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                    "deleted": False,
                    "extracted": False,
                    "restored": False,
                }
            ]
            state["dockerfiles"] = [
                {
                    "index": 1,
                    "label": "part0001",
                    "dockerfile": (model_dir / "dockerfiles" / "Dockerfile.part0001").as_posix(),
                    "tag": "ns/pushmodel:part0001",
                    "included_parts": ["a-part0001.bin"],
                    "built": True,
                    "pushed": True,
                    "pulled": False,
                    "removed": False,
                    "extracted": False,
                }
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                model_name="PushModel",
                docker_namespace="ns",
                max_docker_image_size_bytes=10,
                keep_images=True,
                keep_parts=False,
            )
            commands = []

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.docker_image_exists", return_value=False),
                patch("src.phases.run_docker", side_effect=lambda args: commands.append(args)),
            ):
                main.phase_push_docker(args)

            self.assertEqual(commands[0][0], "build")
            self.assertNotIn(["push", "ns/pushmodel:part0001"], commands)
            saved = main.load_state(model_dir)
            self.assertTrue(saved["parts"][0]["deleted"])

    def test_pull_dockerhub_skips_or_cancels_unpushed_and_repulls_missing_image(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "PullModel"
            state = main.make_initial_state("PullModel", "Org/PullModel", "main", "url", 4, 10)
            state["dockerfiles"] = [
                {
                    "tag": "ns/pullmodel:part0001",
                    "included_parts": [],
                    "pushed": False,
                    "pulled": False,
                },
                {
                    "tag": "ns/pullmodel:part0002",
                    "included_parts": [],
                    "pushed": True,
                    "pulled": True,
                },
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(model_name="PullModel", docker_pull_prefix=None)
            commands = []

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.confirm_unpushed", return_value=(True, None)),
                patch("src.phases.docker_image_exists", return_value=False),
                patch("src.phases.run_docker", side_effect=lambda args: commands.append(args)),
            ):
                main.phase_pull_dockerhub(args)

            self.assertEqual(commands, [["pull", "ns/pullmodel:part0002"]])

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.confirm_unpushed", return_value=(False, None)),
            ):
                with self.assertRaises(RuntimeError):
                    main.phase_pull_dockerhub(args)

    def test_restore_llm_repulls_reextracts_and_rerestores_invalid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "RestoreModel"
            payload = b"abcdef"
            raw_sha = hashlib.sha256(payload).hexdigest()
            part_sha = hashlib.sha256(payload).hexdigest()
            part = {
                "raw_filename": "a.bin",
                "index": 1,
                "part_filename": "a-part0001.bin",
                "path": (model_dir / "parts" / "a-part0001.bin").as_posix(),
                "size_bytes": len(payload),
                "size": "6B",
                "sha256": part_sha,
                "deleted": True,
                "extracted": True,
                "restored": False,
            }
            extracted = model_dir / "extracted" / "parts" / "a-part0001.bin"
            extracted.parent.mkdir(parents=True)
            extracted.write_bytes(payload)
            restored = model_dir / "extracted" / "restored" / "a.bin"
            restored.parent.mkdir(parents=True)
            restored.write_bytes(b"bad")
            state = main.make_initial_state("RestoreModel", "Org/RestoreModel", "main", "url", 10, 10)
            state["raw"] = [
                {
                    "filename": "a.bin",
                    "path": (model_dir / "raw" / "a.bin").as_posix(),
                    "size_bytes": len(payload),
                    "size": "6B",
                    "sha256": raw_sha,
                    "downloaded": True,
                    "split": True,
                    "deleted": True,
                }
            ]
            state["parts"] = [part]
            state["dockerfiles"] = [
                {
                    "index": 1,
                    "label": "part0001",
                    "tag": "ns/restoremodel:part0001",
                    "included_parts": ["a-part0001.bin"],
                    "pushed": True,
                    "pulled": True,
                    "removed": False,
                    "extracted": True,
                }
            ]
            state["restore"] = [
                {
                    "raw_filename": "a.bin",
                    "restored_path": restored.as_posix(),
                    "size_bytes": 3,
                    "size": "3B",
                    "sha256": hashlib.sha256(b"bad").hexdigest(),
                    "restored": True,
                }
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                model_name="RestoreModel",
                docker_pull_prefix=None,
                keep_images=True,
                keep_extracted_parts=True,
            )
            commands = []

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.docker_image_exists", return_value=False),
                patch("src.phases.run_docker", side_effect=lambda args: commands.append(args)),
                patch("src.phases.extract_image_parts") as extract,
            ):
                main.phase_restore_llm(args)

            self.assertEqual(commands, [["pull", "ns/restoremodel:part0001"]])
            extract.assert_not_called()
            self.assertEqual(restored.read_bytes(), payload)

    def test_restore_llm_reextracts_when_extracted_part_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "ExtractModel"
            state = main.make_initial_state("ExtractModel", "Org/ExtractModel", "main", "url", 10, 10)
            state["parts"] = [
                {
                    "raw_filename": "a.bin",
                    "index": 1,
                    "part_filename": "a-part0001.bin",
                    "path": (model_dir / "parts" / "a-part0001.bin").as_posix(),
                    "size_bytes": 1,
                    "size": "1B",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                    "deleted": True,
                    "extracted": True,
                    "restored": False,
                }
            ]
            state["dockerfiles"] = [
                {
                    "index": 1,
                    "label": "part0001",
                    "tag": "ns/extractmodel:part0001",
                    "included_parts": ["a-part0001.bin"],
                    "pushed": True,
                    "pulled": True,
                    "removed": False,
                    "extracted": True,
                }
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                model_name="ExtractModel",
                docker_pull_prefix=None,
                keep_images=True,
                keep_extracted_parts=True,
            )

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.docker_image_exists", return_value=True),
                patch("src.phases.extract_image_parts") as extract,
            ):
                main.phase_restore_llm(args)

            extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
