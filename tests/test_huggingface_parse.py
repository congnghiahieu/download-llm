from __future__ import annotations

import unittest

import main
from src.phases import PHASE_HANDLERS


class HuggingfaceParseTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
