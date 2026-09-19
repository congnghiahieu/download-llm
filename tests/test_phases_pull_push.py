from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class PhasesPullPushTests(unittest.TestCase):
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
            state = main.make_initial_state(
                "PushModel", "Org/PushModel", "main", "url", 4, 10
            )
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
                    "dockerfile": (
                        model_dir / "dockerfiles" / "Dockerfile.part0001"
                    ).as_posix(),
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
                patch(
                    "src.phases.run_docker",
                    side_effect=lambda args: commands.append(args),
                ),
            ):
                main.phase_push_docker(args)

            self.assertEqual(commands[0][0], "build")
            self.assertNotIn(["push", "ns/pushmodel:part0001"], commands)
            saved = main.load_state(model_dir)
            self.assertTrue(saved["parts"][0]["deleted"])

    def test_push_docker_records_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "M"
            state = main.make_initial_state("M", "Org/M", "main", "url", 10**9, 10**9)
            state["parts"] = [
                {
                    "raw_filename": "a.bin",
                    "index": 1,
                    "part_filename": "a-part0001.bin",
                    "path": (model_dir / "parts" / "a-part0001.bin").as_posix(),
                    "size_bytes": 6,
                    "size": "6B",
                    "sha256": hashlib.sha256(b"x").hexdigest(),
                    "deleted": False,
                    "extracted": False,
                    "restored": False,
                }
            ]
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                model_name="M",
                docker_namespace="ns",
                max_docker_image_size_bytes=10**9,
                keep_images=True,
                keep_parts=True,
                model_weights_dir=str(root),
            )
            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.docker_image_exists", return_value=True),
                patch("src.phases.run_docker", return_value=None),
                patch("src.phases.docker_image_size", return_value=12345),
            ):
                main.phase_push_docker(args)
            saved = main.load_state(model_dir)
        self.assertEqual(saved["dockerfiles"][0]["size_bytes"], 12345)
        self.assertEqual(saved["dockerfiles"][0]["size"], "12.35KB")


if __name__ == "__main__":
    unittest.main()
