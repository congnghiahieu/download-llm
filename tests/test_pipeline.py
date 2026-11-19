from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import main
from src.files import restore_raw_file
from src.pipeline import run_models
from src.planner_capacity import ResourceCapacity, measure_resources
from src.planning import Estimate, PhasePlan, make_plan, render_plan
from src.scheduling import DEFAULT_LIMITS, DISK_HEADROOM, Storage
from src.state import is_pull_push_complete, load_state


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = patch("src.planning.WslStorageProbe.detect", return_value=None)
        probe.start()
        self.addCleanup(probe.stop)
        resources = patch(
            "src.planning.measure_resources",
            return_value=ResourceCapacity(16, 16 * 1024**3),
        )
        resources.start()
        self.addCleanup(resources.stop)
        backend = patch("src.planning.docker_uses_containerd_store", return_value=False)
        self.docker_backend = backend.start()
        self.addCleanup(backend.stop)

    def _args(self, root: Path, phases: str, *extra: str):
        return main.parse_args(
            ["--phases", phases, "--model-weights-dir", str(root), *extra]
        )

    def test_plan_report_shows_disk_per_model_and_only_selected_operations(
        self,
    ) -> None:
        model = Mock()
        model.name = "Example"
        model.args = SimpleNamespace(phases=["push_docker"])
        model.state = {"raw": [{"size_bytes": 100}, {"size_bytes": 200}]}
        estimate = Estimate({"docker": "volume"})
        estimate.step("docker", 70, 40, kind="docker_build")
        estimate.step("docker", 50, kind="docker_push")
        for _ in range(2):
            estimate.task("docker_build", docker=20)
        for _ in range(3):
            estimate.task("docker_push", docker=10)
        volume = {"roles": ["docker"], "free_bytes": DISK_HEADROOM + 150}
        plan = PhasePlan([model], {}, [estimate], {"volume": volume}, 1)
        plan.configure(1, {**DEFAULT_LIMITS, "docker_build": 2, "docker_push": 3})

        self.assertEqual(plan.output()["models"][0]["model_size_bytes"], 300)
        self.assertEqual(
            plan.output()["models"][0]["disk"],
            [
                {
                    "roles": ["docker"],
                    "final_additional_bytes": 0,
                    "peak_additional_bytes": 120,
                }
            ],
        )
        self.assertEqual(
            plan.output()["run_task_limits"], {"docker_build": 2, "docker_push": 3}
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            render_plan(plan)
        report = stream.getvalue()
        self.assertTrue(report.startswith("Disk check: Fits\n"))
        self.assertIn("Extra at peak", report)
        self.assertIn("Build Docker images", report)
        self.assertIn("Push Docker images", report)
        self.assertNotIn("Download Hugging Face files", report)

        volume["free_bytes"] = DISK_HEADROOM + 100
        plan.configure(1, plan.limits)
        self.assertEqual(plan.output()["disk_status"], "fits_reduced_concurrency")
        self.assertEqual(
            plan.output()["run_task_limits"], {"docker_build": 1, "docker_push": 1}
        )
        self.assertEqual(
            plan.output()["models"][0]["disk"][0]["peak_additional_bytes"], 90
        )

    def test_resource_probe_respects_affinity_and_cgroup_budgets(self) -> None:
        values = {
            "/proc/meminfo": "MemAvailable: 4096 kB",
            "/proc/self/cgroup": "0::/example",
            "/sys/fs/cgroup/example/cpu.max": "250000 100000",
            "/sys/fs/cgroup/example/memory.max": "4194304",
            "/sys/fs/cgroup/example/memory.current": "1048576",
            "/sys/fs/cgroup/cpu.max": "max 100000",
        }
        with (
            patch(
                "src.planner_capacity._read",
                side_effect=lambda path: values.get(str(path), ""),
            ),
            patch(
                "src.planner_capacity.os.sched_getaffinity", return_value=set(range(8))
            ),
            patch(
                "src.planner_capacity.subprocess.check_output",
                return_value="4 16777216",
            ),
        ):
            self.assertEqual(
                measure_resources(docker=True), ResourceCapacity(2.5, 3 * 1024**2)
            )

    def test_automatic_limits_follow_job_count_and_measured_budgets(self) -> None:
        model = Mock()
        model.args = SimpleNamespace(phases=["push_docker"])
        model.state = {"raw": []}
        estimate = Estimate({"docker": "volume"})
        for _ in range(3):
            estimate.task("docker_build", docker=80)
        estimate.step("docker", 80)
        volume: dict[str, Any] = {
            "roles": ["docker"],
            "free_bytes": DISK_HEADROOM + 250,
        }
        plan = PhasePlan([model], {}, [estimate], {"volume": volume}, 1)
        ample = ResourceCapacity(8, 8 * 1024**3)

        plan.configure_auto(1, ample)
        self.assertEqual(plan.output()["task_limits_source"], "calculated")
        self.assertEqual(
            plan.output()["run_task_limits"], {"docker_build": 3, "docker_push": 0}
        )
        self.assertEqual(volume["peak_additional_bytes"], 240)
        self.assertEqual(plan.output()["remaining_tasks"], {"docker_build": 3})

        volume["free_bytes"] = DISK_HEADROOM + 170
        plan.configure_auto(1, ample)
        self.assertEqual(plan.limits["docker_build"], 2)
        self.assertEqual(volume["peak_additional_bytes"], 160)

        plan.configure_auto(1, ResourceCapacity(8, 2 * 1024**3))
        self.assertEqual(plan.limits["docker_build"], 1)
        plan.configure_auto(1, ResourceCapacity(1, ample.available_memory_bytes))
        self.assertFalse(plan.overlap)
        self.assertEqual(plan.serial_reason, "resources")

        volume["free_bytes"] = None
        plan.configure_auto(1, ample)
        self.assertEqual(plan.limits["docker_build"], 1)
        self.assertEqual(plan.output()["disk_status"], "unknown")

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

    def test_resume_after_cleanup_regenerates_only_unpublished_parts_from_pinned_commit(self) -> None:
        commit = "a" * 40
        digest = "sha256:" + "b" * 64

        def download(repo: str, name: str, revision: str, dest: Path) -> Path:
            self.assertEqual((repo, revision), ("o/m", commit))
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / name
            path.write_bytes(b"abcd")
            return path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "m"
            extra = ("--huggingface-link", "https://huggingface.co/o/m", "--max-part-size", "2B", "--max-docker-image-size", "2B")
            with (
                patch("src.pipeline.model_snapshot", return_value=(commit, [("a.bin", 4)])) as snapshot,
                patch("src.pipeline.download_snapshot_file", side_effect=download) as download_file,
                patch("src.pipeline.build_isolated_image", return_value=2) as build,
                patch("src.pipeline.push_image_digest", return_value=digest) as push,
                patch("src.pipeline.docker_image_exists", return_value=False),
                patch("src.pipeline.remove_local_image"),
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.planning.docker_storage", return_value=root),
            ):
                plan = make_plan(self._args(root, "pull_llm,push_docker", *extra, "--keep-raw", "--keep-parts"))
                run_models(plan.models, plan.max_workers, plan.scheduler())
                state = load_state(model)
                published, pending = state["dockerfiles"]
                pending.update(pushed=False, built=False)
                pending.pop("digest")
                state["publication"]["complete"] = False
                main.save_state(model, state)
                # Simulate a crash after deleting the parts but before updating their flags.
                for part in (model / "parts").iterdir():
                    part.unlink()
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main.execute(main.parse_args([
                        "--cleanup", "--model-name", "m", "--model-weights-dir", tmp,
                        "--delete", "raw", "--yes",
                    ])), 0)
                self.assertTrue(all(not part["deleted"] for part in load_state(model)["parts"]))
                build.reset_mock()
                push.reset_mock()
                download_file.reset_mock()
                plan = make_plan(self._args(root, "pull_llm,push_docker", *extra))
                self.assertEqual(plan.output()["remaining_tasks"]["split"], 1)
                run_models(plan.models, plan.max_workers, plan.scheduler())
                download_file.assert_called_once()
                snapshot.assert_called_once()  # The retry does not resolve the moving branch.
                build.assert_called_once()
                self.assertEqual(build.call_args.args[1]["tag"], pending["tag"])
                push.assert_called_once_with(pending["tag"])
            current = load_state(model)
            self.assertTrue(is_pull_push_complete(current))
            self.assertEqual(current["dockerfiles"][0]["digest"], published["digest"])
            self.assertFalse((model / "parts" / published["included_parts"][0]).exists())

    def test_restore_after_cleanup_reuses_extracted_then_pulls_exact_digest(self) -> None:
        payload = b"xy"
        digest = "sha256:" + "d" * 64
        sha = hashlib.sha256(payload).hexdigest()
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            state = main.make_initial_state("model", "o/m", "main", "https://huggingface.co/o/m", 2, 2)
            state["snapshot"] = {"commit": commit, "files": [["a.bin", 2]]}
            state["publication"] = {"complete": True, "commit": commit}
            state["raw"] = [{"filename": "a.bin", "downloaded": True, "split": True, "size_bytes": 2, "sha256": sha}]
            state["parts"] = [{"part_filename": "a-part0001.bin", "raw_filename": "a.bin", "size_bytes": 2,
                               "sha256": sha, "index": 1, "extracted": True}]
            state["restore"] = [{"raw_filename": "a.bin", "restored": True, "sha256": sha}]
            state["dockerfiles"] = [{"tag": "o/m:part0001", "built": True, "pushed": True, "pulled": True,
                                     "digest": digest, "included_parts": ["a-part0001.bin"]}]
            main.save_state(model, state)
            extracted = model / "extracted/parts/a-part0001.bin"
            restored = model / "extracted/restored/a.bin"
            for path in (extracted, restored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            def clean(categories: str) -> None:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main.execute(main.parse_args([
                        "--cleanup", "--model-name", "model", "--model-weights-dir", tmp,
                        "--delete", categories, "--yes",
                    ])), 0)

            def extract(directory, entry, prefix, *, target_dir):
                (target_dir / extracted.name).write_bytes(payload)

            args = self._args(root, "restore_llm", "--model-name", "model", "--keep-extracted-parts")
            with (
                patch("src.planning.docker_storage", return_value=root),
                patch("src.pipeline.docker_image_exists", return_value=False),
                patch("src.pipeline.run_docker") as docker,
                patch("src.pipeline.extract_image_parts", side_effect=extract),
                patch("src.pipeline.remove_local_image"),
            ):
                clean("restored")
                self.assertFalse(restored.exists())
                self.assertTrue(load_state(model)["restore"][0]["deleted"])
                plan = make_plan(args)
                run_models(plan.models, plan.max_workers, plan.scheduler())
                docker.assert_not_called()
                self.assertEqual(restored.read_bytes(), payload)
                clean("restored,extracted")
                plan = make_plan(args)
                run_models(plan.models, plan.max_workers, plan.scheduler())
                docker.assert_called_once_with(["pull", f"o/m@{digest}"])
            self.assertEqual(restored.read_bytes(), payload)
            current = load_state(model)
            self.assertFalse(current["restore"][0]["deleted"])
            self.assertTrue(is_pull_push_complete(current))

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

    def test_containerd_push_skips_admission_probe_but_build_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(
                root,
                "pull_llm,push_docker",
                "--huggingface-link",
                "https://huggingface.co/o/m",
            )

            def download(repo: str, name: str, revision: str, dest: Path) -> Path:
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / name
                path.write_bytes(b"ab")
                return path

            with (
                patch(
                    "src.pipeline.model_snapshot", return_value=("a" * 40, [("a", 2)])
                ),
                patch("src.pipeline.download_snapshot_file", side_effect=download),
                patch("src.pipeline.build_isolated_image", return_value=2),
                patch(
                    "src.pipeline.push_image_digest", return_value="sha256:" + "b" * 64
                ),
                patch("src.pipeline.remove_local_image"),
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.planning.docker_storage", return_value=root),
            ):
                classic = make_plan(args)
                self.docker_backend.return_value = True
                plan = make_plan(args)
                self.assertEqual(plan.remaining_tasks()["docker_push"], 1)
                self.assertLess(
                    plan.output()["filesystems"][0]["peak_additional_bytes"],
                    classic.output()["filesystems"][0]["peak_additional_bytes"],
                )
                scheduler = plan.scheduler()
                submit = scheduler.submit
                checked = set()

                def record_probe(kind, estimates, work):
                    store = plan.storage["docker"]
                    with patch.object(store, "measure", wraps=store.measure) as measure:
                        future = submit(kind, estimates, work)
                    if measure.called:
                        checked.add(kind)
                    return future

                with patch.object(scheduler, "submit", side_effect=record_probe):
                    run_models(plan.models, 1, scheduler)
                self.assertIn("docker_build", checked)
                self.assertNotIn("docker_push", checked)
                self.assertTrue(is_pull_push_complete(load_state(root / "m")))

    def test_admission_failure_saves_in_flight_push_before_exit(self) -> None:
        lost_probe = Event()
        digest = "sha256:" + "b" * 64

        def download(repo: str, name: str, revision: str, dest: Path) -> Path:
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / name
            path.write_bytes(b"ab")
            return path

        def push(tag: str) -> str:
            self.assertTrue(lost_probe.wait(5))
            return digest

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
                    return_value=("a" * 40, [(name, 2) for name in "abc"]),
                ),
                patch("src.pipeline.download_snapshot_file", side_effect=download),
                patch("src.pipeline.build_isolated_image", return_value=2),
                patch("src.pipeline.push_image_digest", side_effect=push) as pushed,
                patch("src.pipeline.remove_local_image") as cleanup,
                patch("src.planning.planning_cache_dir", return_value=None),
                patch("src.planning.docker_storage", return_value=root),
            ):
                plan = make_plan(args)
                plan.configure(1, {**DEFAULT_LIMITS, "docker_push": 2})
                scheduler = plan.scheduler()
                submit = scheduler.submit
                admitted_push = False

                def submit_or_fail(kind, disk, work):
                    nonlocal admitted_push
                    if kind == "docker_push" and admitted_push:
                        lost_probe.set()
                        raise RuntimeError(
                            "Disk capacity for docker can no longer be verified"
                        )
                    future = submit(kind, disk, work)
                    if kind == "docker_push" and future is not None:
                        admitted_push = True
                    return future

                def verify_saved_before_cleanup(tag: str) -> None:
                    entries = load_state(root / "m")["dockerfiles"]
                    self.assertEqual(
                        next(entry for entry in entries if entry["tag"] == tag)[
                            "digest"
                        ],
                        digest,
                    )

                cleanup.side_effect = verify_saved_before_cleanup
                with (
                    patch.object(scheduler, "submit", side_effect=submit_or_fail),
                    self.assertRaisesRegex(RuntimeError, "Disk capacity for docker"),
                ):
                    run_models(plan.models, 1, scheduler)

                pushed.assert_called_once()
                admitted_tag = pushed.call_args.args[0]
                cleanup.assert_called_once_with(admitted_tag)
                entries = load_state(root / "m")["dockerfiles"]
                self.assertEqual(
                    [entry["tag"] for entry in entries if entry["pushed"]],
                    [admitted_tag],
                )

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
