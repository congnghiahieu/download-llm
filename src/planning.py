from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.constants import (
    PHASE_PULL_DOCKERHUB,
    PHASE_PULL_LLM,
    PHASE_PUSH_DOCKER,
    PHASE_RESTORE_LLM,
)
from src.docker import docker_image_exists
from src.execution import copy_args_for_model, resolve_pull_targets
from src.files import extracted_parts_exist, path_matches, restored_file_valid
from src.huggingface import planning_cache_dir
from src.naming import docker_pull_tag
from src.pipeline import ModelPipeline, prepare_model
from src.scheduling import (
    DEFAULT_LIMITS,
    DISK_HEADROOM,
    Scheduler,
    Storage,
    confirm,
    docker_storage,
)
from src.sizes import human_size
from src.state import effective_base_dir, resolve_model_names


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

    def step(
        self,
        role: str,
        needed: int,
        retained: int = 0,
        *,
        temporary: bool = True,
        kind: str | None = None,
    ) -> None:
        device = self.device_for[role]
        self.serial_peak[device] = max(
            self.serial_peak[device], self.live[device] + needed
        )
        self.live[device] += retained
        if temporary:
            self.possible_temp[device] += retained
        if kind and needed > retained:
            self.scratch[device, kind].append(needed - retained)

    def release(self, role: str, count: int) -> None:
        self.live[self.device_for[role]] -= count

    def keep(self, role: str, count: int) -> None:
        self.retained[self.device_for[role]] += count


def _estimate_send(model: ModelPipeline, calc: Estimate) -> set[str]:
    state, args = model.state, model.args
    entries = state["dockerfiles"] if model.publishing else []
    ready = {part["part_filename"] for part in state["parts"]}
    parts = {part["part_filename"]: part for part in model.inventory}
    remaining = [entry for entry in entries if not entry.get("pushed")]
    added_parts: set[str] = set()
    local_builds: set[str] = set()

    def publish_ready() -> None:
        for entry in remaining[:]:
            if not ready.issuperset(entry["included_parts"]):
                continue
            remaining.remove(entry)
            tag = entry["tag"]
            size = model.image_estimate(entry)
            if not entry.get("built") or not docker_image_exists(tag):
                calc.step(
                    "docker",
                    2 * size,
                    size,
                    temporary=not args.keep_images,
                    kind="docker_build",
                )
                local_builds.add(tag)
            calc.step("docker", size, kind="docker_push")
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
        by_raw: dict[str, list[str]] = defaultdict(list)
        for name, part in parts.items():
            by_raw[part["raw_filename"]].append(name)
        for raw in state["raw"]:
            if raw.get("split"):
                calc.transfer["hf_reused"] += raw["size_bytes"]
                continue
            size = raw["size_bytes"]
            download = model.missing_download_bytes(raw)
            calc.transfer["hf_download"] += download
            calc.transfer["hf_reused"] += size - download
            added_raw = 0
            if not (
                raw.get("downloaded") and path_matches(model.raw_path(raw), size, None)
            ):
                if model.cache_role:
                    calc.step("hf_cache", download, download, temporary=False)
                    calc.keep("hf_cache", download)
                added_raw = size  # a hard-link may work, but copy fallback needs this reservation
                calc.step("weights", size, size, temporary=not args.keep_raw)
                if args.keep_raw:
                    calc.keep("weights", size)
            calc.step(
                "weights",
                size,
                size,
                temporary=model.publishing and not args.keep_parts,
            )
            if not model.publishing or args.keep_parts:
                calc.keep("weights", size)
            names = by_raw[raw["filename"]]
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
    saved = {item["raw_filename"] for item in state["restore"] if item.get("restored")}
    if model.restoring:
        for raw in state["raw"]:
            name = raw["filename"]
            if name not in saved or not restored_file_valid(model.directory, raw):
                pending[name] = raw["size_bytes"]
    ready = {
        name
        for name, part in parts.items()
        if part.get("extracted") and extracted_parts_exist(model.directory, [part])
    }
    new_parts: set[str] = set()

    def restore_ready() -> None:
        for name, size in list(pending.items()):
            if by_raw[name] <= ready:
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
            calc.transfer["docker_download_uncompressed_estimate"] += size
            calc.step("docker", 2 * size, size, temporary=False, kind="docker_pull")
            if retained_image:
                calc.keep("docker", size)
            else:
                calc.pull_backlog[calc.device_for["docker"]].append(size)
        else:
            calc.transfer["docker_reused_uncompressed_estimate"] += size
        if not model.restoring:
            continue
        kept_parts = sum(parts[name]["size_bytes"] for name in missing)
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
    limits: dict[str, int] = field(default_factory=DEFAULT_LIMITS.copy)
    overlap: bool = True
    requested_workers: int = 1

    def configure(self, workers: int, limits: dict[str, int]) -> None:
        if workers < 1 or any(limits[kind] < 1 for kind in DEFAULT_LIMITS):
            raise ValueError("Model and task limits must be positive")
        self.requested_workers = min(workers, len(self.models))
        self.max_workers = self.requested_workers
        self.limits = limits.copy()
        self.overlap = True
        for device, volume in self.volumes.items():
            total = sum(item.retained[device] for item in self.estimates)
            carry = minimum = 0
            for item in self.estimates:
                minimum = max(minimum, carry + item.serial_peak[device])
                carry += item.retained[device]
            volume["final_additional_bytes"] = total
            volume["serial_peak_additional_bytes"] = max(total, minimum)

        def peak(device: str, count: int) -> int:
            extras = sorted(
                (item.possible_temp[device] for item in self.estimates), reverse=True
            )
            scratch = sum(
                sum(
                    sorted(
                        (
                            size
                            for item in self.estimates
                            for size in item.scratch.get((device, kind), [])
                        ),
                        reverse=True,
                    )[:limit]
                )
                for kind, limit in self.limits.items()
            )
            buffered_pulls = sorted(
                (
                    size
                    for item in self.estimates
                    for size in item.pull_backlog.get(device, [])
                ),
                reverse=True,
            )
            window = self.limits["docker_pull"] + self.limits["extract_restore"]
            return max(
                self.volumes[device]["serial_peak_additional_bytes"],
                self.volumes[device]["final_additional_bytes"]
                + sum(extras[:count])
                + scratch
                + sum(buffered_pulls[:window]),
            )

        def fits(device: str, additional: int) -> bool:
            available = self.volumes[device]["free_bytes"]
            return available is None or available >= additional + DISK_HEADROOM

        while self.max_workers > 1 and not all(
            fits(dev, peak(dev, self.max_workers)) for dev in self.volumes
        ):
            self.max_workers -= 1
        if not all(fits(dev, peak(dev, self.max_workers)) for dev in self.volumes):
            self.overlap = False
            self.max_workers = 1
        for device, volume in self.volumes.items():
            final = volume["final_additional_bytes"]
            serial = volume["serial_peak_additional_bytes"]
            volume["peak_additional_bytes"] = (
                peak(device, self.max_workers) if self.overlap else serial
            )
            if volume["free_bytes"] is None:
                status = "unknown"
            elif not fits(device, final):
                status = "insufficient_final"
            elif not fits(device, serial):
                status = "insufficient_temporary"
            elif not fits(device, peak(device, self.requested_workers)):
                status = "fits_reduced_concurrency"
            else:
                status = "fits"
            volume["status"] = status

    def output(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": model.name,
                    "phases": model.args.phases,
                    "revision": model.state.get("snapshot", {}).get("commit"),
                    "network_bytes": dict(estimate.transfer),
                }
                for model, estimate in zip(self.models, self.estimates)
            ],
            "requested_model_workers": self.requested_workers,
            "model_workers": self.max_workers,
            "task_limits": self.limits,
            "task_overlap": self.overlap,
            "disk_headroom_bytes": DISK_HEADROOM,
            "filesystems": list(self.volumes.values()),
            "notes": [
                "Disk peaks including Docker layers/cache are estimates, not guaranteed bounds.",
                "Hugging Face raw files reserve space for a copy even when the stdlib backend may hard-link from cache.",
                "Docker network estimates are uncompressed sizes; registry layer reuse and wire sizes are not known.",
                "Concurrency is global. Disk checks may reduce it at runtime.",
                "Python streams file I/O in chunks; Docker memory and disk I/O vary by daemon and workload.",
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
    cache = planning_cache_dir() if PHASE_PULL_LLM in phases else None
    if cache is not None:
        storage["hf_cache"] = Storage(cache.absolute())
    if set(phases) - {PHASE_PULL_LLM}:
        storage["docker"] = Storage(docker_storage(), docker=True)
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
    models = [
        prepare_model(
            copy_args_for_model(args, name, link), base / name, cache is not None
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
        if free is not None:
            volume["free_bytes"] = min(volume["free_bytes"], free)
    estimates = [_model_estimate(model, devices) for model in models]
    workers = getattr(args, "max_workers", None) or len(models)
    plan = PhasePlan(models, storage, estimates, volumes, workers)
    plan.configure(workers, DEFAULT_LIMITS)
    return plan


def render_plan(
    plan: PhasePlan, output: str = "table", *, to_stderr: bool = False
) -> None:
    stream = sys.stderr if to_stderr else sys.stdout
    info = plan.output()
    if output == "json":
        print(json.dumps(info, indent=2), file=stream)
        return
    print(f"Models: {', '.join(model.name for model in plan.models)}", file=stream)
    for model in info["models"]:
        network = ", ".join(
            f"{name}={human_size(size)}"
            for name, size in model["network_bytes"].items()
        )
        if network:
            print(f"  {model['name']} network: {network}", file=stream)
    print(
        f"Model workers: {plan.max_workers}/{plan.requested_workers}, task overlap: {plan.overlap}",
        file=stream,
    )
    print(
        "Task limits: "
        + ", ".join(f"{kind}={limit}" for kind, limit in plan.limits.items()),
        file=stream,
    )
    print(f"Disk (headroom per filesystem: {human_size(DISK_HEADROOM)})", file=stream)
    print(
        f"{'Filesystem roles':25} {'Free':>12} {'Final +':>12} {'Serial peak +':>14} {'Peak +':>12}  Status",
        file=stream,
    )
    for volume in plan.volumes.values():
        free = (
            "unknown"
            if volume["free_bytes"] is None
            else human_size(volume["free_bytes"])
        )
        print(
            f"{', '.join(volume['roles']):25} {free:>12} "
            f"{human_size(volume['final_additional_bytes']):>12} "
            f"{human_size(volume['serial_peak_additional_bytes']):>14} "
            f"{human_size(volume['peak_additional_bytes']):>12}  {volume['status']}",
            file=stream,
        )
    for note in info["notes"]:
        print(note, file=stream)
