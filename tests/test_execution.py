from __future__ import annotations

import argparse
import os
import unittest
from unittest.mock import patch

import main


class MultiExecutionTests(unittest.TestCase):
    def _make_pull_item(self, name: str, link: str) -> argparse.Namespace:
        return argparse.Namespace(
            phases=["pull_llm"],
            model_name=name,
            model_names=[name],
            huggingface_link=link,
            huggingface_links=[link],
            revision="main",
            docker_namespace="ns",
            docker_pull_prefix=None,
            max_part_size="5GB",
            max_docker_image_size="5GB",
            keep_raw=False,
            keep_parts=False,
            keep_images=False,
            keep_extracted_parts=False,
        )

    def _make_restore_item(self, name: str) -> argparse.Namespace:
        return argparse.Namespace(
            phases=["pull_dockerhub"],
            model_name=name,
            model_names=[name],
            huggingface_link=None,
            huggingface_links=[],
            revision=None,
            docker_namespace="ns",
            docker_pull_prefix=None,
            max_part_size="5GB",
            max_docker_image_size="5GB",
            keep_raw=False,
            keep_parts=False,
            keep_images=False,
            keep_extracted_parts=False,
        )

    def test_sequential_multi_is_best_effort_with_retry_report(self) -> None:
        calls = []

        def fake_phase(args: argparse.Namespace) -> None:
            calls.append(args.model_name)
            if args.model_name == "A":
                raise ConnectionError("network down")

        work_items = [
            self._make_pull_item("A", "https://huggingface.co/Qwen/A"),
            self._make_pull_item("B", "https://huggingface.co/Org/B"),
        ]
        with (
            patch.dict(main.PHASE_HANDLERS, {"pull_llm": fake_phase}),
            self.assertRaises(RuntimeError) as ctx,
        ):
            main.run_sequential_models(work_items, argparse.Namespace(max_workers=1))
        self.assertIn("A", str(ctx.exception))
        self.assertEqual(calls, ["A", "B"])
        retry = main.build_retry_command(work_items[0])
        self.assertIn("https://huggingface.co/Qwen/A", retry)
        self.assertIn("--phases pull_llm", retry)

    def test_parallel_multi_reports_failed_model_with_retry(self) -> None:
        calls = []

        def fake_phase(args: argparse.Namespace) -> None:
            calls.append(args.model_name)
            if args.model_name == "B":
                raise TimeoutError("lost connection")

        work_items = [self._make_restore_item("A"), self._make_restore_item("B")]
        with (
            patch.dict(main.PHASE_HANDLERS, {"pull_dockerhub": fake_phase}),
            self.assertRaises(RuntimeError) as ctx,
        ):
            main.run_parallel_models(work_items, argparse.Namespace(max_workers=2))
        self.assertIn("B", str(ctx.exception))
        self.assertEqual(sorted(calls), ["A", "B"])

    def test_execute_fails_fast_when_docker_unavailable(self) -> None:
        args = argparse.Namespace(
            phases=["push_docker"],
            model_names=["Qwen3.6-27B"],
            model_name="Qwen3.6-27B",
        )
        with (
            patch(
                "src.cli.ensure_docker_available",
                side_effect=RuntimeError("Docker is not available"),
            ),
            patch.dict(main.PHASE_HANDLERS, {"push_docker": lambda a: None}),
            self.assertRaises(RuntimeError),
        ):
            main.execute(args)

    def test_detect_command_prefix_uses_uv_inside_venv(self) -> None:
        with (
            patch("sys.executable", "/repo/.venv/bin/python"),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(main.detect_command_prefix(), "uv run python main.py")
        with (
            patch.dict(os.environ, {"VIRTUAL_ENV": "/repo/.venv"}, clear=True),
            patch("sys.executable", "/usr/bin/python3"),
        ):
            self.assertEqual(main.detect_command_prefix(), "uv run python main.py")

    def test_detect_command_prefix_defaults_python3(self) -> None:
        with (
            patch("sys.executable", "/usr/bin/python3"),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(main.detect_command_prefix(), "python3 main.py")

    def test_retry_command_uses_detected_prefix(self) -> None:
        item = argparse.Namespace(
            phases=["pull_llm"],
            model_name=None,
            model_names=[],
            huggingface_link="https://huggingface.co/Qwen/A",
            huggingface_links=["https://huggingface.co/Qwen/A"],
            revision=None,
            docker_namespace=None,
            docker_pull_prefix=None,
            max_part_size=None,
            max_docker_image_size=None,
            model_weights_dir="model_weights",
        )
        with (
            patch("sys.executable", "/usr/bin/python3"),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertTrue(
                main.build_retry_command(item).startswith("python3 main.py")
            )


if __name__ == "__main__":
    unittest.main()
