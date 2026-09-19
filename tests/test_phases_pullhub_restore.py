from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class PhasesPullhubRestoreTests(unittest.TestCase):
    def test_pull_dockerhub_skips_or_cancels_unpushed_and_repulls_missing_image(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "PullModel"
            state = main.make_initial_state(
                "PullModel", "Org/PullModel", "main", "url", 4, 10
            )
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
                patch(
                    "src.phases.run_docker",
                    side_effect=lambda args: commands.append(args),
                ),
            ):
                main.phase_pull_dockerhub(args)

            self.assertEqual(commands, [["pull", "ns/pullmodel:part0002"]])

            with (
                patch("src.phases.BASE_DIR", root),
                patch("src.phases.confirm_unpushed", return_value=(False, None)),
                self.assertRaises(RuntimeError),
            ):
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
            state = main.make_initial_state(
                "RestoreModel", "Org/RestoreModel", "main", "url", 10, 10
            )
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
                patch(
                    "src.phases.run_docker",
                    side_effect=lambda args: commands.append(args),
                ),
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
            state = main.make_initial_state(
                "ExtractModel", "Org/ExtractModel", "main", "url", 10, 10
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
