from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from src.cli import parse_args
from src.delivery import Receiver, file_stamp, run_with_retry
from src.delivery_remote import META
from src.pipeline import run_models
from src.planning import DeliveryDiskForecast, _backlog_peak, make_plan
from src.scheduling import DISK_HEADROOM
from src.state import expected_restored_path, make_initial_state, save_state


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / "weights"
        self.destination = self.root / "destination's space"

    def manifest(self, model, files, *, staged=()):
        directory = self.weights / model
        directory.mkdir(parents=True)
        state = make_initial_state(
            model, f"o/{model}", "main", f"https://huggingface.co/o/{model}", 50, 50
        )
        for i, (name, content) in enumerate(files.items()):
            digest = hashlib.sha256(content).hexdigest()
            part = f"part-{i}"
            state["raw"].append(
                {
                    "filename": name,
                    "size_bytes": len(content),
                    "sha256": digest,
                    "downloaded": True,
                    "split": True,
                }
            )
            state["parts"].append(
                {
                    "part_filename": part,
                    "raw_filename": name,
                    "size_bytes": len(content),
                    "sha256": digest,
                    "index": 1,
                }
            )
            state["dockerfiles"].append(
                {
                    "tag": f"o/{model}:{i}",
                    "built": True,
                    "pushed": True,
                    "included_parts": [part],
                }
            )
            if name in staged:
                file = directory / expected_restored_path(name)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(content)
        save_state(directory, state)
        return directory

    def args(self, names="m", phases="deliver_llm", *extra):
        return parse_args(
            [
                "--phases",
                phases,
                "--model-name",
                names,
                "--model-weights-dir",
                str(self.weights),
                "--ssh-host",
                "local",
                "--destination-path",
                str(self.destination),
                *extra,
            ]
        )

    def run_delivery(self, args):
        with patch(
            "src.planning.docker_storage",
            side_effect=AssertionError("must not call Docker"),
        ):
            plan = make_plan(args)
            self.assertTrue(plan.authorize_run())
            run_models(plan.models, plan.max_workers, plan.scheduler())

    @unittest.skipUnless(shutil.which("rsync"), "real rsync required")
    def test_real_rsync_stages_and_resumes_then_publishes_without_clobbering(self):
        ssh = self.root / "ssh"
        ssh.write_text(
            '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\ncase "$1" in\n'
            "  -o|-l|-p|-i) shift 2 ;;\n  -*) shift ;;\n  *) shift; break ;;\n"
            'esac\ndone\nexec /bin/sh -c "$*"\n'
        )
        ssh.chmod(0o700)
        a, b = "folder /a's file", "weights.bin"
        directory = self.manifest("m", {a: b"first", b: b"second"}, staged=[a])
        stage = self.destination / "m.transferring"
        final = self.destination / "m"
        with patch.dict(os.environ, {"PATH": f"{self.root}:{os.environ['PATH']}"}):
            with self.assertRaisesRegex(RuntimeError, r"1 file\(s\) not verified"):
                self.run_delivery(self.args())
            self.assertEqual((stage / a).read_bytes(), b"first")
            self.assertFalse((directory / expected_restored_path(a)).exists())
            self.assertFalse(final.exists())

            local = directory / expected_restored_path(b)
            local.write_bytes(b"wrong!")
            with self.assertRaisesRegex(RuntimeError, "failed SHA-256 verification"):
                self.run_delivery(self.args())
            self.assertEqual(local.read_bytes(), b"wrong!")
            self.assertFalse(final.exists())

            # Simulate an interrupted rsync with a partial already at the receiver.
            partial = (
                stage / META / (hashlib.sha256(b.encode()).hexdigest() + ".partial")
            )
            partial.write_bytes(b"sec")
            local.write_bytes(b"second")
            self.run_delivery(self.args("m", "deliver_llm", "--keep-local"))
            self.assertEqual((final / a).read_bytes(), b"first")
            self.assertEqual((final / b).read_bytes(), b"second")
            self.assertFalse((final / META).exists())
            self.assertFalse(stage.exists())
            self.assertTrue(local.exists())
            self.run_delivery(self.args())
            self.assertFalse(local.exists())

            with (
                patch("src.planning.docker_storage", return_value=self.root),
                patch("src.pipeline.run_docker") as docker,
            ):
                plan = make_plan(self.args("m", "restore_llm,deliver_llm"))
                run_models(plan.models, plan.max_workers, plan.scheduler())
            docker.assert_not_called()
            self.assertFalse(
                local.exists(), "combined resume reuses its verified receipts"
            )

            (final / b).write_bytes(b"wrong!")
            with self.assertRaisesRegex(RuntimeError, "does not match manifest"):
                self.run_delivery(self.args())
            self.assertEqual((final / b).read_bytes(), b"wrong!")

    def test_worker_chooses_largest_ready_file_across_models(self):
        self.manifest("m", {"small": b"a", "large": b"aaaa"}, staged=["small", "large"])
        self.manifest("other", {"largest": b"aaaaaa"}, staged=["largest"])
        with (
            patch.object(Receiver, "inspect", side_effect=lambda: (False, set())),
            patch.object(Receiver, "file", return_value=True) as copy,
            patch.object(Receiver, "publish") as publish,
        ):
            self.run_delivery(self.args("m,other", "deliver_llm", "--keep-local"))
            self.assertEqual(
                [call.args[1]["filename"] for call in copy.call_args_list],
                ["largest", "large", "small"],
            )
            self.assertEqual(publish.call_count, 2)

    def test_cli_requires_delivery_destination_and_scopes_keep_local(self):
        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--phases", "deliver_llm", "--model-name", "m"])
        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args(["--phases", "restore_llm", "--keep-local"])

    def test_transfer_failure_does_not_stop_restore(self):
        directory = self.manifest("m", {"first": b"a", "second": b"bb"})
        delivery_started = Event()

        def failed_copy(*_args):
            delivery_started.set()
            raise RuntimeError("link down")

        def pull(command):
            if command[1].endswith(":1"):
                self.assertTrue(
                    delivery_started.wait(5),
                    "first delivery starts before the next pull finishes",
                )

        def extract(_directory, entry, _prefix, *, target_dir):
            for part in entry["included_parts"]:
                (target_dir / part).write_bytes({"part-0": b"a", "part-1": b"bb"}[part])

        with (
            patch("src.planning.docker_storage", return_value=self.root),
            patch("src.pipeline.run_docker", side_effect=pull),
            patch("src.pipeline.extract_image_parts", side_effect=extract),
            patch("src.pipeline.remove_local_image"),
            patch.object(Receiver, "inspect", return_value=(False, set())),
            patch.object(Receiver, "file", side_effect=failed_copy),
            self.assertLogs("download_llm", level="INFO") as logs,
        ):
            plan = make_plan(self.args("m", "pull_dockerhub,restore_llm,deliver_llm"))
            with self.assertRaisesRegex(RuntimeError, "link down"):
                run_models(
                    plan.models,
                    plan.max_workers,
                    plan.scheduler(),
                    DeliveryDiskForecast(plan),
                )
        self.assertIn("Restore [m]: done (2/2 files)", "\n".join(logs.output))
        self.assertIn("Delivery [m]: incomplete", "\n".join(logs.output))
        self.assertEqual(
            (directory / expected_restored_path("first")).read_bytes(), b"a"
        )
        self.assertEqual(
            (directory / expected_restored_path("second")).read_bytes(), b"bb"
        )

    def test_disk_stops_restore_without_waiting_for_transfer(self):
        directory = self.manifest(
            "m", {"first": b"a", "second": b"bb"}, staged=["first"]
        )
        disk_reported = Event()

        def copy(path, *_args):
            self.assertTrue(
                disk_reported.wait(3),
                "disk admission must stop before transfer completes",
            )
            return file_stamp(path)

        def report(*args, **kwargs):
            if kwargs.get("blocked"):
                disk_reported.set()
            return forecast_report(*args, **kwargs)

        with (
            patch("src.planning.docker_storage", return_value=self.root),
            patch.object(Receiver, "inspect", return_value=(False, set())),
            patch.object(Receiver, "file", side_effect=copy),
            patch("src.scheduling.Storage.measure", return_value=(1, DISK_HEADROOM)),
            patch("src.pipeline.run_docker") as docker,
            self.assertLogs("download_llm", level="INFO") as logs,
        ):
            plan = make_plan(self.args("m", "restore_llm,deliver_llm"))
            self.assertTrue(plan.authorize_run())
            forecast = DeliveryDiskForecast(plan)
            forecast_report = forecast.report
            with (
                patch.object(forecast, "report", side_effect=report),
                self.assertRaisesRegex(RuntimeError, "disk budget"),
            ):
                run_models(plan.models, plan.max_workers, plan.scheduler(), forecast)
        self.assertIn("Restore remaining: 1 file(s)", "\n".join(logs.output))
        self.assertIn("Disk admission stopped", "\n".join(logs.output))
        self.assertFalse(
            (directory / expected_restored_path("first")).exists(),
            "verified staged file still drains",
        )
        self.assertFalse((directory / expected_restored_path("second")).exists())
        docker.assert_not_called()

    def test_retry_only_transient_network_failures(self):
        connection = subprocess.CalledProcessError(
            255, "ssh", stderr=b"Connection reset by peer"
        )
        auth = subprocess.CalledProcessError(
            255, "ssh", stderr=b"Permission denied (publickey)"
        )
        success = subprocess.CompletedProcess("ssh", 0, b"done", b"")
        with (
            patch("src.delivery.time.sleep"),
            patch(
                "src.delivery.subprocess.run",
                side_effect=[connection, connection, success],
            ) as run,
        ):
            self.assertEqual(run_with_retry(["ssh"]), "done")
            self.assertEqual(run.call_count, 3)
        with patch("src.delivery.subprocess.run", side_effect=auth) as run:
            with self.assertRaisesRegex(RuntimeError, "Permission denied"):
                run_with_retry(["ssh"])
            run.assert_called_once()

    def test_forecast_frees_only_a_whole_completed_file(self):
        self.assertEqual(_backlog_peak([(9, True), (2, True)], [], 1, 1), 11)
        self.assertEqual(_backlog_peak([(9, True), (12, True)], [], 1, 1), 12)
        self.assertEqual(_backlog_peak([(9, False), (12, False)], [], 1, 1), 21)
        self.assertEqual(_backlog_peak([(9, True)], [12], 1, 2), 0)
