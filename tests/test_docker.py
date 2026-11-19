from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from scripts import src_image
from src.docker import push_image_digest
from src.run_logging import run_logging
from src.scheduling import docker_storage


class DockerUtilsTests(unittest.TestCase):
    def test_docker_desktop_storage_is_not_treated_as_local_through_unix_socket(
        self,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {"DOCKER_HOST": "unix:///var/run/docker.sock", "DOCKER_CONTEXT": ""},
            ),
            patch(
                "src.scheduling._docker_output",
                return_value="Docker Desktop (containerized)\n/var/lib/docker",
            ),
        ):
            self.assertIsNone(docker_storage())

    def test_docker_output_and_exit_status_are_logged_and_remain_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "download-llm.log"
            docker = root / "docker"
            docker.write_text(
                "#!/bin/sh\nprintf 'layer retrying\\n'\n"
                "printf 'write /var/lib/docker: no space left on device\\n' >&2\nexit 1\n",
                encoding="utf-8",
            )
            docker.chmod(0o700)
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("src.run_logging.LOG_PATH", log_path),
                patch.dict(
                    os.environ,
                    {"PATH": str(root) + os.pathsep + os.environ.get("PATH", "")},
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                run_logging(),
                self.assertRaises(subprocess.CalledProcessError) as failure,
            ):
                push_image_digest("test/image:part001")
            self.assertIn("layer retrying", stdout.getvalue())
            self.assertIn("no space left on device", stderr.getvalue())
            self.assertIn("no space left on device", failure.exception.stderr)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("layer retrying", log)
            self.assertIn("no space left on device", log)
            self.assertIn("exit=1", log)
            self.assertEqual(len(set(re.findall(r"command=([0-9a-f]+)", log))), 1)

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
