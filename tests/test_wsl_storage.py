from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from src.planning import Estimate, PhasePlan
from src.scheduling import DEFAULT_LIMITS, DISK_HEADROOM, Device, Scheduler, Storage
from src.wsl_storage import (
    DockerWslStorage,
    WslStorageProbe,
    _registry_entries,
    _windows_path,
)


class FixedStorage(Storage):
    def __init__(self, device: Device, free: int) -> None:
        super().__init__(None)
        self.device = device
        self.free = free

    def measure(self) -> tuple[Device, int]:
        return self.device, self.free


class WslStorageTests(unittest.TestCase):
    def test_registry_keeps_distribution_and_backing_path_paired(self) -> None:
        entries = _registry_entries(
            "HKEY_CURRENT_USER\\Software\\Lxss\\{1}\n"
            "    DistributionName    REG_SZ    Ubuntu-24.04\n"
            "    BasePath    REG_SZ    D:\\WSL Images\\Ubuntu\n"
            "    VhdFileName    REG_SZ    ubuntu.vhdx\n"
            "HKEY_CURRENT_USER\\Software\\Lxss\\{2}\n"
            "    BasePath    REG_SZ    \\\\?\\C:\\Docker\\wsl\\main\n"
            "    DistributionName    REG_SZ    docker-desktop\n"
        )
        self.assertEqual(_windows_path(entries["ubuntu-24.04"]["BasePath"]).drive, "D:")
        self.assertEqual(entries["ubuntu-24.04"]["VhdFileName"], "ubuntu.vhdx")
        self.assertEqual(
            _windows_path(entries["docker-desktop"]["BasePath"]).drive, "C:"
        )

    def test_docker_guest_probe_has_distinct_device_and_reports_bytes(self) -> None:
        with patch(
            "src.wsl_storage._output",
            side_effect=["Ubuntu-24.04\ndocker-desktop", "2080 852714008"],
        ) as output:
            storage = DockerWslStorage(Path("/var/lib/docker"))
            self.assertEqual(storage.measure(), ("docker-wsl:2080", 852714008 * 1024))
        self.assertEqual(output.call_args.args[-1], "/var/lib/docker")

    def test_unavailable_docker_guest_stays_unknown_without_starting_distro(
        self,
    ) -> None:
        with patch("src.wsl_storage._output", return_value="Ubuntu-24.04") as output:
            storage = DockerWslStorage(Path("/var/lib/docker"))
            self.assertIsNone(storage.measure())
        output.assert_called_once()
        self.assertIn("not running", str(storage.measurement_error))

    def test_configured_docker_disk_does_not_fall_back_to_old_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp)
            (settings / "settings-store.json").write_text(
                json.dumps({"dataFolder": {"value": r"D:\Docker Data"}}),
                encoding="utf-8",
            )
            with (
                patch.object(
                    WslStorageProbe,
                    "_entry",
                    return_value={"BasePath": r"C:\Docker\wsl\main"},
                ),
                patch("src.wsl_storage._output", return_value=r"C:\Profile"),
                patch(
                    "src.wsl_storage._linux_path",
                    side_effect=[settings, Path("/mnt/d/Docker Data")],
                ),
                patch(
                    "src.wsl_storage._vhd_storage",
                    side_effect=ValueError("disk missing"),
                ) as verify,
            ):
                backing = WslStorageProbe().docker_host()
            self.assertIn(
                Path("/mnt/d/Docker Data/disk/docker_data.vhdx"),
                verify.call_args.args[0],
            )
            self.assertEqual(verify.call_args.args[1].drive, "D:")
            self.assertIsNone(backing.measure())
            self.assertIn("disk missing", str(backing.measurement_error))

    def test_planner_combines_host_demand_without_merging_guest_volumes(self) -> None:
        volumes = {
            "ubuntu": {"roles": ["weights"], "free_bytes": DISK_HEADROOM + 100},
            "engine": {"roles": ["docker"], "free_bytes": DISK_HEADROOM + 100},
            "C": {
                "roles": ["weights_host", "docker_host"],
                "free_bytes": DISK_HEADROOM + 120,
            },
        }
        estimate = Estimate(
            {
                "weights": "ubuntu",
                "docker": "engine",
                "weights_host": "C",
                "docker_host": "C",
            }
        )
        for role in ("weights", "docker"):
            estimate.step(role, 70, 70, temporary=False)
            estimate.keep(role, 70)
        plan = PhasePlan([Mock()], {}, [estimate], volumes, 1)
        plan.configure(1, DEFAULT_LIMITS)
        self.assertEqual(volumes["C"]["final_additional_bytes"], 140)
        self.assertEqual(volumes["C"]["status"], "insufficient_final")
        self.assertEqual(volumes["ubuntu"]["status"], "fits")
        self.assertEqual(volumes["engine"]["status"], "fits")

    def test_scheduler_reserves_one_shared_host_budget_for_different_tasks(
        self,
    ) -> None:
        host = FixedStorage(10, DISK_HEADROOM + 100)
        storage: dict[str, Storage] = {
            "weights": FixedStorage(1, DISK_HEADROOM + 1000),
            "docker": FixedStorage("docker-wsl:1", DISK_HEADROOM + 1000),
            "weights_host": host,
            "docker_host": host,
        }
        finish = Event()
        with closing(
            Scheduler(DEFAULT_LIMITS, storage, overlap=True, unknown=set())
        ) as scheduler:
            try:
                first = scheduler.submit("hf_download", {"weights": 60}, finish.wait)
                self.assertIsNotNone(first)
                self.assertIsNone(
                    scheduler.submit("docker_build", {"docker": 50}, lambda: None)
                )
                self.assertEqual(scheduler.reserved[10], 60)
            finally:
                finish.set()

    def test_different_windows_drives_have_separate_runtime_budgets(self) -> None:
        storage: dict[str, Storage] = {
            "weights": FixedStorage(1, DISK_HEADROOM + 1000),
            "docker": FixedStorage("docker-wsl:1", DISK_HEADROOM + 1000),
            "weights_host": FixedStorage(10, DISK_HEADROOM + 100),
            "docker_host": FixedStorage(11, DISK_HEADROOM + 100),
        }
        finish = Event()
        with closing(
            Scheduler(DEFAULT_LIMITS, storage, overlap=True, unknown=set())
        ) as scheduler:
            try:
                self.assertIsNotNone(
                    scheduler.submit("hf_download", {"weights": 60}, finish.wait)
                )
                self.assertIsNotNone(
                    scheduler.submit("docker_build", {"docker": 50}, lambda: None)
                )
            finally:
                finish.set()
