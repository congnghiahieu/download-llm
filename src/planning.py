from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from src.constants import (
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
)
from src.docker import docker_image_exists, docker_uses_containerd_store
from src.execution import copy_args_for_model, resolve_pull_targets
from src.files import extracted_parts_exist, restored_file_valid
from src.huggingface import planning_cache_dir
from src.naming import docker_pull_tag
from src.pipeline import ModelPipeline, _pushed, prepare_model
from src.planner_capacity import ResourceCapacity, measure_resources
from src.scheduling import (
    DISK_HEADROOM,
    DockerDesktopRoot,
    Scheduler,
    Storage,
    confirm,
    docker_storage,
    storage_constraints,
)
from src.sizes import human_size
from src.state import effective_base_dir, resolve_model_names
from src.wsl_storage import DockerWslStorage, WindowsBackingStorage, WslStorageProbe

TASK_LABELS = {
    "hf_download": "Download Hugging Face files",
    "split": "Split files into parts",
    "docker_build": "Build Docker images",
    "docker_push": "Push Docker images",
    "docker_pull": "Pull Docker images",
    "extract_restore": "Extract images / restore files (shared)",
}
# Estimates per admitted task; Docker may use more internally (e.g. parallel layers).
_MIB = 1024**2
_TASK_COST = {
    "hf_download": (1, 128 * _MIB),
    "split": (1, 128 * _MIB),
    "docker_build": (2, 1024 * _MIB),
    "docker_push": (2, 512 * _MIB),
    "docker_pull": (1, 512 * _MIB),
    "extract_restore": (1, 256 * _MIB),
}
_PHASE_TASKS = {
    PHASE_PULL_LLM: ("hf_download", "split"),
    PHASE_PUSH_DOCKER: ("docker_build", "docker_push"),
    PHASE_PULL_DOCKERHUB: ("docker_pull",),
    PHASE_RESTORE_LLM: ("docker_pull", "extract_restore"),
}
_DISK_STATUS = {
    "fits": "Fits",
    "fits_reduced_concurrency": "Fits with reduced concurrency",
    "insufficient_final": "Not enough space for retained data",
    "insufficient_temporary": "Not enough working space, even in serial",
    "unknown": "Unknown capacity; confirmation required",
}
_STORAGE_LABELS = {
    "weights": "Model files",
    "hf_cache": "Hugging Face cache",
    "docker": "Docker storage",
    "weights_host": "Model files",
    "hf_cache_host": "Hugging Face cache",
    "docker_host": "Docker",
}


@dataclass
class Estimate:
    device_for: dict[str, str]
    retained: Counter[str] = field(default_factory=Counter)
    live: Counter[str] = field(default_factory=Counter)
    serial_peak: Counter[str] = field(default_factory=Counter)
    possible_temp: Counter[str] = field(default_factory=Counter)
    scratch: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pull_backlog: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    transfer: Counter[str] = field(default_factory=Counter)
    jobs: dict[str, list[dict[str, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def devices(self, role: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.device_for[name]
                for name in storage_constraints(role, self.device_for)
            )
        )

    def task(self, kind: str, **bytes_by_role: int) -> None:
        reservation: Counter[str] = Counter()
        for role, size in bytes_by_role.items():
            for device in self.devices(role):
                reservation[device] += size
        self.jobs[kind].append(dict(reservation))

    def step(
        self,
        role: str,
        needed: int,
        retained: int = 0,
        *,
        temporary: bool = True,
        kind: str | None = None,
    ) -> None:
        for device in self.devices(role):
            self.serial_peak[device] = max(
                self.serial_peak[device], self.live[device] + needed
            )
            self.live[device] += retained
            if temporary:
                self.possible_temp[device] += retained
            if kind and needed > retained:
                self.scratch[device, kind].append(needed - retained)

    def release(self, role: str, count: int) -> None:
        for device in self.devices(role):
            self.live[device] -= count

    def keep(self, role: str, count: int) -> None:
        for device in self.devices(role):
            self.retained[device] += count


def _estimate_send(model: ModelPipeline, calc: Estimate) -> set[str]:
    if not (model.downloading or model.publishing):
        return set()
    state, args = model.state, model.args
    entries = state["dockerfiles"] if model.publishing else []
    ready = {part["part_filename"] for part in state["parts"] if model.part_available(part)}
    parts = {part["part_filename"]: part for part in model.inventory}
    pinned = bool(state.get("snapshot"))
    remaining = [entry for entry in entries if not _pushed(entry, pinned)]
    existing_builds = {entry["tag"] for entry in remaining if entry.get("built") and docker_image_exists(entry["tag"])}
    required = model.required_send_parts(existing_builds)
    added_parts: set[str] = set()
    local_builds: set[str] = set()

    def publish_ready() -> None:
        for entry in remaining[:]:
            tag = entry["tag"]
            if tag not in existing_builds and not ready.issuperset(entry["included_parts"]):
                continue
            remaining.remove(entry)
            size = model.image_estimate(entry)
            if tag not in existing_builds:
                calc.task("docker_build", docker=2 * size)
                calc.step(
                    "docker",
                    2 * size,
                    size,
                    temporary=not args.keep_images,
                    kind="docker_build",
                )
                local_builds.add(tag)
            demand = model.push_disk_demand(size)
            calc.task("docker_push", **demand)
            for role, required in demand.items():
                calc.step(role, required, kind="docker_push")
            calc.transfer["docker_upload_uncompressed_estimate"] += size
            if args.keep_images and tag in local_builds:
                calc.keep("docker", size)
            # Otherwise carry the built size as potential Docker cache, not required output.
            if not args.keep_parts:
                for name in entry["included_parts"]:
                    if name in added_parts:
                        calc.release("weights", parts[name]["size_bytes"])

    publish_ready()
    if model.downloading:
        for raw in state["raw"]:
            split = model.needs_split(raw, ready, required)
            if not split and not args.keep_raw:
                calc.transfer["hf_reused"] += raw["size_bytes"]
                continue
            size = raw["size_bytes"]
            download = model.missing_download_bytes(raw)
            calc.transfer["hf_download"] += download
            calc.transfer["hf_reused"] += size - download
            added_raw = 0
            if not model.raw_available(raw):
                if model.cache_role:
                    calc.step("hf_cache", download, download, temporary=False)
                    calc.keep("hf_cache", download)
                demand = {"hf_cache": download} if model.cache_role else {}
                calc.task("hf_download", weights=size, **demand)
                added_raw = size  # a hard-link may work, but copy fallback needs this reservation
                calc.step("weights", size, size, temporary=not args.keep_raw)
                if args.keep_raw:
                    calc.keep("weights", size)
            if not split:
                continue
            names = model.missing_split_parts(raw, ready, required)
            split_size = sum(parts[name]["size_bytes"] for name in names)
            calc.task("split", weights=split_size)
            calc.step(
                "weights",
                split_size,
                split_size,
                temporary=model.publishing and not args.keep_parts,
            )
            if not model.publishing or args.keep_parts:
                calc.keep("weights", split_size)
            added_parts.update(names)
            ready.update(names)
            if not args.keep_raw:
                calc.release("weights", added_raw)
            publish_ready()
    return local_builds if args.keep_images else set()


def _estimate_receive(
    model: ModelPipeline, calc: Estimate, local_builds: set[str]
) -> None:
    state, args = model.state, model.args
    parts = {part["part_filename"]: part for part in model.inventory}
    by_raw: dict[str, set[str]] = defaultdict(set)
    for name, part in parts.items():
        by_raw[part["raw_filename"]].add(name)
    pending: dict[str, int] = {}
    if model.restoring:
        for raw in state["raw"]:
            name = raw["filename"]
            if not raw.get("sha256") or not restored_file_valid(model.directory, raw):
                pending[name] = raw["size_bytes"]
    ready = {
        part["part_filename"]
        for part in state["parts"]
        if part["raw_filename"] in pending and part.get("sha256")
        and part["part_filename"] in parts and extracted_parts_exist(model.directory, [part])
    }
    new_parts: set[str] = set()

    def restore_ready() -> None:
        for name, size in list(pending.items()):
            if by_raw[name] <= ready:
                calc.task("extract_restore", weights=size)
                calc.step("weights", size, size, temporary=False)
                calc.keep("weights", size)
                del pending[name]
                if not args.keep_extracted_parts:
                    calc.release(
                        "weights",
                        sum(parts[p]["size_bytes"] for p in by_raw[name] & new_parts),
                    )

    restore_ready()
    for entry in state["dockerfiles"]:
        missing = {
            name
            for name in entry["included_parts"]
            if parts[name]["raw_filename"] in pending and name not in ready
        }
        if model.restoring and not missing:
            continue
        size = model.image_estimate(entry)
        reference = docker_pull_tag(entry, args.docker_pull_prefix)
        local = (
            (entry.get("pulled") or entry.get("digest"))
            and docker_image_exists(reference)
        ) or (entry["tag"] in local_builds and not args.docker_pull_prefix)
        retained_image = args.keep_images or not model.restoring
        if not local:
            if not model.publishing or _pushed(entry, bool(state.get("snapshot"))):
                model.require_pull_digest(entry)
            calc.transfer["docker_download_uncompressed_estimate"] += size
            calc.task("docker_pull", docker=2 * size)
            calc.step("docker", 2 * size, size, temporary=False, kind="docker_pull")
            if retained_image:
                calc.keep("docker", size)
            else:
                for device in calc.devices("docker"):
                    calc.pull_backlog[device].append(size)
        else:
            calc.transfer["docker_reused_uncompressed_estimate"] += size
        if not model.restoring:
            continue
        kept_parts = sum(parts[name]["size_bytes"] for name in missing)
        calc.task("extract_restore", weights=model.image_parts_size(entry))
        calc.step(
            "weights",
            model.image_parts_size(entry),
            kept_parts,
            temporary=not args.keep_extracted_parts,
            kind="extract_restore",
        )
        if args.keep_extracted_parts:
            calc.keep("weights", kept_parts)
        ready.update(missing)
        new_parts.update(missing)
        if not local and not retained_image:
            calc.release("docker", size)
        restore_ready()
    if pending:
        raise ValueError(
            f"{model.name}: manifest does not contain all parts needed for restore"
        )


def _model_estimate(model: ModelPipeline, devices: dict[str, str]) -> Estimate:
    calc = Estimate(devices)
    local_builds = (
        _estimate_send(model, calc) if model.downloading or model.publishing else set()
    )
    if model.receiving:
        _estimate_receive(model, calc, local_builds)
    return calc


@dataclass
class PhasePlan:
    models: list[ModelPipeline]
    storage: dict[str, Storage]
    estimates: list[Estimate]
    volumes: dict[str, dict[str, Any]]
    max_workers: int
    limits: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(TASK_LABELS, 1)
    )
    overlap: bool = True
    requested_workers: int = 1
    resources: ResourceCapacity | None = None
    task_limits_source: str = "configured"
    serial_reason: str | None = None

    def _peak(
        self,
        device: str,
        estimates: list[Estimate],
        workers: int,
        serial: int,
        final: int,
        limits: dict[str, int] | None = None,
    ) -> int:
        caps = self.limits if limits is None else limits
        extras = sorted(
            (item.possible_temp[device] for item in estimates), reverse=True
        )
        scratch = sum(
            sum(
                sorted(
                    (
                        size
                        for item in estimates
                        for size in item.scratch.get((device, kind), [])
                    ),
                    reverse=True,
                )[:limit]
            )
            for kind, limit in caps.items()
        )
        buffered = sorted(
            (size for item in estimates for size in item.pull_backlog.get(device, [])),
            reverse=True,
        )
        window = caps["docker_pull"] + caps["extract_restore"]
        reservations = sum(
            sum(
                sorted(
                    (
                        job.get(device, 0)
                        for item in estimates
                        for job in item.jobs.get(kind, [])
                    ),
                    reverse=True,
                )[:limit]
            )
            for kind, limit in caps.items()
        )
        return max(
            serial,
            final + sum(extras[:workers]) + scratch + sum(buffered[:window]),
            reservations,
        )

    def _batch_peak(
        self, device: str, workers: int, limits: dict[str, int] | None = None
    ) -> int:
        volume = self.volumes[device]
        return self._peak(
            device,
            self.estimates,
            workers,
            volume["serial_peak_additional_bytes"],
            volume["final_additional_bytes"],
            limits,
        )

    def _fits(self, device: str, additional: int) -> bool:
        available = self.volumes[device]["free_bytes"]
        return available is None or available >= additional + DISK_HEADROOM

    def configure(
        self, workers: int, limits: dict[str, int], *, allow_overlap: bool = True
    ) -> None:
        if workers < 1 or any(limits[kind] < 1 for kind in TASK_LABELS):
            raise ValueError("Model and task limits must be positive")
        self.requested_workers = min(workers, len(self.models))
        self.max_workers = self.requested_workers if allow_overlap else 1
        self.limits = limits.copy()
        self.task_limits_source = "configured"
        self.overlap = allow_overlap
        self.serial_reason = None if allow_overlap else "resources"
        for device, volume in self.volumes.items():
            total = sum(item.retained[device] for item in self.estimates)
            carry = minimum = 0
            for item in self.estimates:
                minimum = max(minimum, carry + item.serial_peak[device])
                carry += item.retained[device]
            volume["final_additional_bytes"] = total
            volume["serial_peak_additional_bytes"] = max(total, minimum)

        while self.max_workers > 1 and not all(
            self._fits(dev, self._batch_peak(dev, self.max_workers))
            for dev in self.volumes
        ):
            self.max_workers -= 1
        if self.overlap and not all(
            self._fits(dev, self._batch_peak(dev, self.max_workers))
            for dev in self.volumes
        ):
            self.overlap = False
            self.max_workers = 1
            self.serial_reason = "disk"
        for device, volume in self.volumes.items():
            final = volume["final_additional_bytes"]
            serial = volume["serial_peak_additional_bytes"]
            volume["peak_additional_bytes"] = (
                self._batch_peak(device, self.max_workers) if self.overlap else serial
            )
            if volume["free_bytes"] is None:
                status = "unknown"
            elif not self._fits(device, final):
                status = "insufficient_final"
            elif not self._fits(device, serial):
                status = "insufficient_temporary"
            elif not self._fits(
                device, self._batch_peak(device, self.requested_workers)
            ):
                status = "fits_reduced_concurrency"
            else:
                status = "fits"
            volume["status"] = status

    def remaining_tasks(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for estimate in self.estimates:
            counts.update({kind: len(jobs) for kind, jobs in estimate.jobs.items()})
        return counts

    def configure_auto(self, workers: int, resources: ResourceCapacity) -> None:
        self.resources = resources
        limits = dict.fromkeys(TASK_LABELS, 1)
        pending = self.remaining_tasks()

        def fits_resources(caps: dict[str, int]) -> bool:
            if not pending:
                return True
            cpu, ram = resources.cpu_threads, resources.available_memory_bytes
            if cpu is None or ram is None:
                return False
            cpus = memory = 0
            for kind, count in pending.items():
                task_cpu, task_memory = _TASK_COST[kind]
                slots = min(caps[kind], count)
                cpus += task_cpu * slots
                memory += task_memory * slots
            return cpus <= cpu and memory <= max(0, ram - max(1024 * _MIB, ram // 5))

        allow_overlap = fits_resources(limits)
        self.configure(workers, limits, allow_overlap=allow_overlap)
        # Keep the model count chosen at one slot per operation. Only grow the
        # per-operation caps if the measured budgets still fit that schedule.
        disk_known = all(
            volume["free_bytes"] is not None for volume in self.volumes.values()
        )
        if self.overlap and disk_known:
            growing = True
            while growing:
                growing = False
                for kind in TASK_LABELS:
                    if limits[kind] >= pending[kind]:
                        continue
                    candidate = limits | {kind: limits[kind] + 1}
                    if fits_resources(candidate) and all(
                        self._fits(
                            device,
                            self._batch_peak(device, self.max_workers, candidate),
                        )
                        for device in self.volumes
                    ):
                        limits = candidate
                        growing = True
            self.configure(workers, limits, allow_overlap=allow_overlap)
        self.task_limits_source = "calculated"

    def selected_task_limits(self) -> dict[str, int]:
        tasks = {
            task
            for model in self.models
            for phase in model.args.phases
            for task in _PHASE_TASKS[phase]
        }
        return {kind: self.limits[kind] for kind in TASK_LABELS if kind in tasks}

    def _model_output(self, model: ModelPipeline, estimate: Estimate) -> dict[str, Any]:
        sizes = [raw.get("size_bytes") for raw in model.state["raw"]]
        disk = []
        for device, volume in self.volumes.items():
            final = estimate.retained[device]
            serial_peak = max(final, estimate.serial_peak[device])
            row = {
                "roles": volume["roles"],
                "final_additional_bytes": final,
                "peak_additional_bytes": self._peak(
                    device, [estimate], 1, serial_peak, final
                )
                if self.overlap
                else serial_peak,
            }
            if drive := volume.get("windows_drive"):
                row["windows_drive"] = drive
            disk.append(row)
        return {
            "name": model.name,
            "phases": model.args.phases,
            "revision": model.state.get("snapshot", {}).get("commit"),
            "model_size_bytes": sum(sizes)
            if sizes and all(isinstance(size, int) for size in sizes)
            else None,
            "disk": disk,
            "network_bytes": dict(estimate.transfer),
        }

    def output(self) -> dict[str, Any]:
        pending = self.remaining_tasks()
        statuses = {volume["status"] for volume in self.volumes.values()}
        status = next(
            (
                key
                for key in (
                    "insufficient_final",
                    "insufficient_temporary",
                    "unknown",
                    "fits_reduced_concurrency",
                )
                if key in statuses
            ),
            "fits",
        )
        return {
            "models": [
                self._model_output(model, estimate)
                for model, estimate in zip(self.models, self.estimates)
            ],
            "requested_model_workers": self.requested_workers,
            "model_workers": self.max_workers,
            "task_limits": self.limits,
            "run_task_limits": {
                task: min(pending[task], limit if self.overlap else 1)
                for task, limit in self.selected_task_limits().items()
            },
            "task_limits_source": self.task_limits_source,
            "remaining_tasks": dict(pending),
            "resource_capacity": {
                "cpu_threads": self.resources.cpu_threads,
                "available_memory_bytes": self.resources.available_memory_bytes,
            }
            if self.resources is not None
            else None,
            "task_overlap": self.overlap,
            "serial_reason": self.serial_reason,
            "disk_status": status,
            "disk_headroom_bytes": DISK_HEADROOM,
            "filesystems": list(self.volumes.values()),
            "notes": [
                "Disk peaks including Docker layers/cache are estimates, not guaranteed bounds.",
                "Hugging Face raw files reserve space for a copy even when the stdlib backend may hard-link from cache.",
                "Docker network estimates are uncompressed sizes; registry layer reuse and wire sizes are not known.",
                "Calculated limits use pending jobs and measured CPU, RAM, and disk, with estimated resource costs per operation.",
                "Network throughput is not benchmarked; runtime disk checks may reduce actual concurrency.",
            ],
        }

    def authorize_run(self) -> bool:
        blocked = [
            f"{', '.join(volume['roles'])}: {volume['status']}"
            for volume in self.volumes.values()
            if volume["status"].startswith("insufficient_")
        ]
        if blocked:
            print(
                f"Execution blocked by disk preflight: {'; '.join(blocked)}",
                file=sys.stderr,
            )
            return False
        unknown = [
            role
            for volume in self.volumes.values()
            if volume["status"] == "unknown"
            for role in volume["roles"]
        ]
        if unknown and not confirm(
            f"Warning: disk capacity is unknown for {', '.join(unknown)}."
        ):
            print(
                "Execution blocked: unknown disk capacity requires terminal confirmation.",
                file=sys.stderr,
            )
            return False
        return True

    def scheduler(self) -> Scheduler:
        return Scheduler(
            self.limits,
            self.storage,
            overlap=self.overlap,
            unknown={
                role
                for volume in self.volumes.values()
                if volume["status"] == "unknown"
                for role in volume["roles"]
            },
        )


def make_plan(args: argparse.Namespace, *, allow_sudo: bool = False) -> PhasePlan:
    base = effective_base_dir(args)
    phases = args.phases
    if not phases:
        raise ValueError("Select at least one phase")
    if len(phases) != len(set(phases)) or phases != sorted(
        phases,
        key=(
            PHASE_PULL_LLM,
            PHASE_PUSH_DOCKER,
            PHASE_PULL_DOCKERHUB,
            PHASE_RESTORE_LLM,
        ).index,
    ):
        raise ValueError("Select each phase once in pipeline order")
    if PHASE_PULL_LLM in phases and PHASE_PUSH_DOCKER not in phases and len(phases) > 1:
        raise ValueError(
            "A pipeline using pull_llm and restore requires push_docker between them"
        )
    storage = {"weights": Storage(base)}
    wsl = WslStorageProbe.detect()
    cache = planning_cache_dir() if PHASE_PULL_LLM in phases else None
    if cache is not None:
        storage["hf_cache"] = Storage(cache.absolute())
    if set(phases) - {PHASE_PULL_LLM}:
        location = docker_storage(include_desktop=wsl is not None)
        if isinstance(location, DockerDesktopRoot):
            assert wsl is not None
            storage["docker"] = DockerWslStorage(location.path)
            storage["docker_host"] = wsl.docker_host()
        else:
            storage["docker"] = Storage(location, docker=True)
    if wsl is not None:
        for role, store in list(storage.items()):
            if role not in {"weights", "hf_cache", "docker"} or isinstance(
                store, DockerWslStorage
            ):
                continue
            if store.path is not None:
                host = wsl.guest_host(store.path)
                if host is not None:
                    storage[f"{role}_host"] = host
    if PHASE_PULL_LLM in phases:
        targets = [(name, link) for link, name in resolve_pull_targets(args)]
    else:
        names = resolve_model_names(args, ",".join(phases), base_dir=base)
        targets = [(name, None) for name in dict.fromkeys(names)]
    directories = [(base / name).resolve() for name, _ in targets]
    if len(directories) != len(set(directories)):
        raise ValueError(
            "Multiple selected models resolve to the same weights directory; run with distinct model names"
        )
    containerd_push = PHASE_PUSH_DOCKER in phases and docker_uses_containerd_store()
    models = [
        prepare_model(
            copy_args_for_model(args, name, link),
            base / name,
            cache is not None,
            containerd_push=containerd_push,
        )
        for name, link in targets
    ]
    if not models:
        raise ValueError("No models selected")
    if PHASE_PUSH_DOCKER in phases:
        tags = [
            entry["tag"] for model in models for entry in model.state["dockerfiles"]
        ]
        if len(tags) != len(set(tags)):
            raise ValueError("Selected models have conflicting Docker image tags")
    if allow_sudo and "docker" in storage:
        storage["docker"].authorize_probe()

    devices: dict[str, str] = {}
    volumes: dict[str, dict[str, Any]] = {}
    for role, store in storage.items():
        value = store.measure()
        dev, free = (
            (str(value[0]), value[1])
            if value is not None
            else (f"unknown:{role}", None)
        )
        devices[role] = dev
        volume = volumes.setdefault(dev, {"roles": [], "paths": [], "free_bytes": free})
        volume["roles"].append(role)
        volume["paths"].append(str(store.path) if store.path is not None else None)
        if isinstance(store, WindowsBackingStorage):
            volume["windows_drive"] = store.windows_drive
        if value is None and store.measurement_error:
            volume["measurement_error"] = store.measurement_error
        if free is not None:
            volume["free_bytes"] = min(volume["free_bytes"], free)
    estimates = [_model_estimate(model, devices) for model in models]
    workers = getattr(args, "max_workers", None) or len(models)
    plan = PhasePlan(models, storage, estimates, volumes, workers)
    plan.configure_auto(workers, measure_resources(docker="docker" in storage))
    return plan


def render_plan(
    plan: PhasePlan, output: str = "table", *, to_stderr: bool = False
) -> None:
    stream = sys.stderr if to_stderr else sys.stdout
    info = plan.output()
    if output == "json":
        print(json.dumps(info, indent=2), file=stream)
        return
    print(f"Disk check: {_DISK_STATUS[info['disk_status']]}", file=stream)
    print(
        f"Reserved free space: {human_size(DISK_HEADROOM)} per filesystem.", file=stream
    )
    rows = []
    for volume in plan.volumes.values():
        rows.append(
            (
                _storage_name(volume),
                _format_bytes(volume["free_bytes"]),
                human_size(volume["final_additional_bytes"]),
                human_size(volume["peak_additional_bytes"]),
                _DISK_STATUS[volume["status"]],
            )
        )
    _print_table(
        stream,
        ("Filesystem", "Free now", "Extra kept", "Extra at peak", "Result"),
        rows,
        numeric=(1, 2, 3),
    )
    for volume in plan.volumes.values():
        if reason := volume.get("measurement_error"):
            print(f"  {_storage_name(volume)}: {reason}", file=stream)

    print("\nDisk by model", file=stream)
    rows = []
    for model in info["models"]:
        for index, volume in enumerate(model["disk"]):
            rows.append(
                (
                    model["name"] if index == 0 else "",
                    _format_bytes(model["model_size_bytes"]) if index == 0 else "",
                    _storage_name(volume),
                    human_size(volume["final_additional_bytes"]),
                    human_size(volume["peak_additional_bytes"]),
                )
            )
    _print_table(
        stream,
        ("Model", "Original size", "Filesystem", "Extra kept", "Extra at peak"),
        rows,
        numeric=(1, 3, 4),
    )
    print(
        "Extra kept is new space still in use at the end; extra at peak is the most new space needed during the run.",
        file=stream,
    )
    print(
        "Existing data is already counted in free space. Docker cache and peak figures are estimates.",
        file=stream,
    )
    if any(volume.get("windows_drive") for volume in plan.volumes.values()):
        print(
            "Windows backing space checks the same writes at the host level. Do not add it to guest figures.",
            file=stream,
        )
    if len(plan.models) > 1:
        print(
            "Per-model peaks assume one model at a time. Use the first table to check space for the full run.",
            file=stream,
        )

    print(
        f"\nConcurrency for this run: {plan.max_workers} of {plan.requested_workers} requested models",
        file=stream,
    )
    if not plan.overlap:
        reason = (
            "CPU/RAM budget" if plan.serial_reason == "resources" else "Disk budget"
        )
        print(
            f"{reason} requires one operation at a time, with no overlap.", file=stream
        )
    if resources := info["resource_capacity"]:
        cpu = resources["cpu_threads"]
        cpu_label = "Unknown" if cpu is None else f"{cpu:g}"
        print(
            f"Capacity checked: {cpu_label} CPU threads, {_format_bytes(resources['available_memory_bytes'])} RAM available.",
            file=stream,
        )
    _print_table(
        stream,
        ("Operation", "Jobs left", "Max at once"),
        [
            (TASK_LABELS[task], str(info["remaining_tasks"].get(task, 0)), str(limit))
            for task, limit in info["run_task_limits"].items()
        ],
        numeric=(1, 2),
    )
    source = (
        "Calculated from pending jobs and available resources"
        if plan.task_limits_source == "calculated"
        else "Configured in this plan"
    )
    print(
        f"{source}. Caps are shared across models; per-job costs are estimates and network speed was not benchmarked.",
        file=stream,
    )


def _format_bytes(value: int | None) -> str:
    return "Unknown" if value is None else human_size(value)


def _storage_name(volume: dict[str, Any]) -> str:
    roles = volume["roles"]
    if drive := volume.get("windows_drive"):
        return f"Windows {drive}" + (" (shared)" if len(roles) > 1 else "")
    return " + ".join(
        f"Windows drive for {_STORAGE_LABELS.get(role, role)}"
        if role.endswith("_host")
        else _STORAGE_LABELS.get(role, role)
        for role in roles
    )


def _print_table(
    stream: TextIO,
    headers: tuple[str, ...],
    rows: Sequence[tuple[str, ...]],
    *,
    numeric: tuple[int, ...] = (),
) -> None:
    widths = [
        max(len(header), max((len(row[i]) for row in rows), default=0))
        for i, header in enumerate(headers)
    ]
    for index, row in enumerate([headers, *rows]):
        cells = (
            value.rjust(widths[i]) if i in numeric else value.ljust(widths[i])
            for i, value in enumerate(row)
        )
        print("  ".join(cells).rstrip(), file=stream)
        if index == 0:
            print("  ".join("-" * width for width in widths), file=stream)
