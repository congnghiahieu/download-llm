from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class StateListingTests(unittest.TestCase):
    def test_phase_status_strict_flags(self) -> None:
        state = {
            "raw": [
                {"filename": "a.bin", "downloaded": True, "split": True},
                {"filename": "b.bin", "downloaded": True, "split": False},
            ],
            "parts": [],
            "dockerfiles": [
                {"built": True, "pushed": True, "pulled": True},
                {"built": True, "pushed": False, "pulled": False},
            ],
            "restore": [{"raw_filename": "a.bin", "restored": True}],
        }
        status = main.phase_status(state)
        self.assertEqual(
            status,
            {
                "pull_llm": False,
                "push_docker": False,
                "pull_dockerhub": False,
                "restore_llm": False,
            },
        )
        state["raw"][1]["split"] = True
        state["dockerfiles"][1].update({"built": True, "pushed": True, "pulled": True})
        state["restore"].append({"raw_filename": "b.bin", "restored": True})
        self.assertEqual(
            main.phase_status(state),
            {
                "pull_llm": True,
                "push_docker": True,
                "pull_dockerhub": True,
                "restore_llm": True,
            },
        )

    def test_summarize_model_sizes_and_unknown_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "M"
            state = main.make_initial_state("M", "Org/M", "main", "url", 10, 10)
            state["raw"] = [
                {"filename": "a.bin", "size_bytes": 1000, "downloaded": True},
                {"filename": "b.bin", "downloaded": False},
            ]
            state["dockerfiles"] = [
                {"tag": "ns/m:part0001", "size_bytes": 5000, "size": "5.00KB"},
                {"tag": "ns/m:part0002"},
            ]
            main.save_state(model_dir, state)
            summary = main.summarize_model(model_dir)
        self.assertEqual(summary["raw_files"], 2)
        self.assertEqual(summary["raw_size_bytes"], 1000)
        self.assertEqual(summary["images"], 2)
        self.assertEqual(summary["image_size_bytes"], 5000)
        self.assertEqual(summary["unknown_images"], 1)
        self.assertIn("unknown", main.format_image_size(summary))

    def test_list_models_flag_and_output_validation(self) -> None:
        args = main.parse_args(["--list-models"])
        self.assertTrue(args.list_models)
        self.assertEqual(args.phases, [])
        self.assertEqual(args.output, "table")
        with self.assertRaises(SystemExit):
            main.parse_args(["--list-models", "--phases", "pull_llm"])
        with self.assertRaises(SystemExit):
            main.parse_args(["--output", "json"])

    def test_execute_list_models_custom_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "M"
            state = main.make_initial_state("M", "Org/M", "main", "url", 10, 10)
            main.save_state(model_dir, state)
            args = argparse.Namespace(
                list_models=True, output="json", model_weights_dir=str(root)
            )
            with patch("sys.stdout", io.StringIO()) as stdout:
                self.assertEqual(main.execute(args), 0)
            payload = json.loads(stdout.getvalue())
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["model"], "M")
        self.assertFalse(payload[0]["status"]["pull_llm"])


if __name__ == "__main__":
    unittest.main()
