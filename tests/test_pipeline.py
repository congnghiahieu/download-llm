from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from unittest.mock import patch

import main
from src.files import restore_raw_file
from src.pipeline import run_models
from src.planning import Estimate, PhasePlan, make_plan
from src.scheduling import DEFAULT_LIMITS, DISK_HEADROOM, Storage
from src.state import is_pull_push_complete, load_state


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = patch("src.planning.WslStorageProbe.detect", return_value=None)
        probe.start()
        self.addCleanup(probe.stop)

    def _args(self, root: Path, phases: str, *extra: str):
        return main.parse_args(
            ["--phases", phases, "--model-weights-dir", str(root), *extra]
        )

    def test_send_pipeline_starts_upload_before_last_file_and_overlaps_build(
        self,
    ) -> None:
        commit = "a" * 40
        push_started, second_built = Event(), Event()

        def download(repo: str, name: str, revision: str, dest: Path) -> Path:
            self.assertEqual(revision, commit)
            if name == "b":
                self.assertTrue(
                    push_started.wait(3), "push must start while b is waiting"
                )
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / name
            path.write_bytes(b"ab" if name == "a" else b"cd")
            return path

        def build(directory, entry, parts):
            if entry["label"] == "part0002":
                second_built.set()
            return 2

        def push(tag: str) -> str:
            if ":part0001-" in tag:
                push_started.set()
                self.assertTrue(
                    second_built.wait(3),
                    "second image must build while first image pushes",
                )
            return "sha256:" + "b" * 64

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(
                root,
                "pull_llm,push_docker",
                "--huggingface-link",
                "https://huggingface.co/o/m",
                "--max-part-size",
                "2B",
                "--max-docker-image-size",
                "2B",
            )
            with (
                patch(
                    "src.pipeline.model_snapshot",
                    return_value=(commit, [("a", 2), ("b", 2)]),
                ),
                patch("src.pipeline.download_snapshot_file", side_effect=download),
                patch("src.pipeline.build_isolated_image", side_effect=build),
                patch("src.pipeline.push_image_digest", side_effect=push),
                patch("src.pipeline.remove_local_image"),
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.planning.docker_storage", return_value=root),
            ):
                plan = make_plan(args)
                self.assertFalse((root / "m").exists())
                run_models(plan.models, plan.max_workers, plan.scheduler())
            state = load_state(root / "m")
            self.assertTrue(is_pull_push_complete(state))
            self.assertEqual(len(state["dockerfiles"]), 2)
            self.assertTrue(all(part["deleted"] for part in state["parts"]))

    def test_receive_pipeline_restores_ready_file_before_unrelated_pull(self) -> None:
        first_restored = Event()
        data = {"a-1": b"xx", "b-1": b"yy"}

        def sha(payload: bytes) -> str:
            return hashlib.sha256(payload).hexdigest()

        def pull(command):
            if command[1].endswith("b"):
                self.assertTrue(
                    first_restored.wait(3), "restore a before pulling b finishes"
                )

        def extract(directory, entry, prefix, *, target_dir):
            for part in entry["included_parts"]:
                (target_dir / part).write_bytes(data[part])

        def restore(directory, raw, parts):
            result = restore_raw_file(directory, raw, parts)
            if raw["filename"] == "a":
                first_restored.set()
            return result

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            state = main.make_initial_state(
                "model", "o/m", "main", "https://huggingface.co/o/m", 2, 2
            )
            for filename in ("a", "b"):
                part = f"{filename}-1"
                state["raw"].append(
                    {
                        "filename": filename,
                        "downloaded": True,
                        "split": True,
                        "size_bytes": 2,
                        "sha256": sha(data[part]),
                    }
                )
                state["parts"].append(
                    {
                        "part_filename": part,
                        "raw_filename": filename,
                        "size_bytes": 2,
                        "sha256": sha(data[part]),
                        "index": 1,
                    }
                )
                state["dockerfiles"].append(
                    {
                        "tag": f"o/m:{filename}",
                        "label": filename,
                        "built": True,
                        "pushed": True,
                        "included_parts": [part],
                    }
                )
            main.save_state(model, state)
            args = self._args(
                root, "pull_dockerhub,restore_llm", "--model-name", "model"
            )
            with (
                patch("src.planning.docker_storage", return_value=root),
                patch("src.pipeline.run_docker", side_effect=pull),
                patch("src.pipeline.extract_image_parts", side_effect=extract),
                patch("src.pipeline.restore_raw_file", side_effect=restore),
                patch("src.pipeline.remove_local_image"),
            ):
                plan = make_plan(args)
                run_models(plan.models, plan.max_workers, plan.scheduler())
            self.assertEqual(
                (model / "extracted" / "restored" / "a").read_bytes(), b"xx"
            )
            self.assertEqual(
                (model / "extracted" / "restored" / "b").read_bytes(), b"yy"
            )
            self.assertTrue(main.phase_status(load_state(model))["restore_llm"])

    def test_disk_plan_distinguishes_final_and_temporary_space(self) -> None:
        dummy = unittest.mock.Mock()
        estimate = Estimate({"weights": "vol"})
        estimate.step("weights", 100, 100, temporary=False)
        estimate.keep("weights", 100)
        estimate.step("weights", 40, 0, kind="extract_restore")
        volume = {"roles": ["weights"], "free_bytes": DISK_HEADROOM + 99}
        plan = PhasePlan([dummy], {}, [estimate], {"vol": volume}, 1)
        plan.configure(1, DEFAULT_LIMITS)
        self.assertEqual(volume["status"], "insufficient_final")
        volume["free_bytes"] = DISK_HEADROOM + 130
        plan.configure(1, DEFAULT_LIMITS)
        self.assertEqual(volume["status"], "insufficient_temporary")

    def test_disk_plan_can_reduce_concurrency_without_blocking_final_output(
        self,
    ) -> None:
        device = {"weights": "vol"}
        estimate = Estimate(device)
        estimate.step("weights", 50, 50)
        estimate.step("weights", 50, 50, temporary=False)
        estimate.keep("weights", 50)
        volume = {"roles": ["weights"], "free_bytes": DISK_HEADROOM + 160}
        plan = PhasePlan(
            [unittest.mock.Mock(), unittest.mock.Mock()],
            {},
            [estimate, estimate],
            {"vol": volume},
            2,
        )
        plan.configure(2, DEFAULT_LIMITS)
        self.assertEqual(volume["status"], "fits_reduced_concurrency")
        self.assertEqual(plan.max_workers, 1)

    def test_sender_resumes_without_redownload_or_repush(self) -> None:
        def download(repo: str, name: str, revision: str, dest: Path) -> Path:
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / name
            path.write_bytes(b"ab")
            return path

        def push(tag: str) -> str:
            if ":part0002-" in tag:
                raise ConnectionError("simulated registry failure")
            return "sha256:" + "b" * 64

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(
                root,
                "pull_llm,push_docker",
                "--huggingface-link",
                "https://huggingface.co/o/m",
                "--max-part-size",
                "2B",
                "--max-docker-image-size",
                "2B",
            )
            with (
                patch(
                    "src.pipeline.model_snapshot",
                    return_value=("a" * 40, [("a", 2), ("b", 2)]),
                ) as snapshot,
                patch(
                    "src.pipeline.download_snapshot_file", side_effect=download
                ) as fetch,
                patch("src.pipeline.build_isolated_image", return_value=2),
                patch("src.pipeline.docker_image_exists", return_value=True),
                patch("src.pipeline.push_image_digest", side_effect=push) as push_mock,
                patch("src.pipeline.remove_local_image"),
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.planning.docker_storage", return_value=root),
            ):
                args.max_workers = 1
                plan = make_plan(args)
                # Serialize jobs to make the simulated failure occur after the first push.
                plan.overlap = False
                with self.assertRaisesRegex(RuntimeError, "simulated registry failure"):
                    run_models(plan.models, plan.max_workers, plan.scheduler())
                first = load_state(root / "m")
                self.assertFalse(is_pull_push_complete(first))
                self.assertTrue(first["dockerfiles"][0]["pushed"])
                self.assertFalse(first["dockerfiles"][1]["pushed"])
                push_mock.reset_mock(side_effect=True)
                push_mock.return_value = "sha256:" + "c" * 64
                plan = make_plan(args)
                run_models(plan.models, plan.max_workers, plan.scheduler())
                self.assertEqual(fetch.call_count, 2)
                snapshot.assert_called_once()
                push_mock.assert_called_once_with(first["dockerfiles"][1]["tag"])
                self.assertTrue(is_pull_push_complete(load_state(root / "m")))

    def test_unknown_disk_blocks_execution_without_terminal(self) -> None:
        plan = PhasePlan(
            [],
            {"docker": Storage(None)},
            [],
            {"docker": {"roles": ["docker"], "status": "unknown"}},
            1,
        )
        output = io.StringIO()
        with (
            patch("src.scheduling.sys.stdin.isatty", return_value=False),
            redirect_stderr(output),
        ):
            self.assertFalse(plan.authorize_run())
        self.assertIn("Execution blocked", output.getvalue())
        self.assertIn("Warning: disk capacity is unknown", output.getvalue())

    def test_json_preview_writes_no_model_state_or_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model_weights"
            log = root / "logs" / "test.log"
            args = self._args(
                root,
                "pull_llm",
                "--plan",
                "--output",
                "json",
                "--huggingface-link",
                "https://huggingface.co/o/m",
            )
            output = io.StringIO()
            with (
                redirect_stdout(output),
                patch(
                    "src.pipeline.model_snapshot",
                    return_value=("a" * 40, [("config.json", 5)]),
                ),
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.run_logging.LOG_PATH", log),
            ):
                self.assertEqual(main.execute(args), 0)
            self.assertEqual(
                json.loads(output.getvalue())["models"][0]["network_bytes"][
                    "hf_download"
                ],
                5,
            )
            self.assertFalse(root.exists())
