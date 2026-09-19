from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import main
from scripts import src_image


class DockerUtilsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
