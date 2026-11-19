from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PureWindowsPath

from src.scheduling import Device, Storage

_DOCKER_DISTRO = "docker-desktop"
_LXSS = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss"
_DRIVE = re.compile(r"^[a-zA-Z]:$")


def _output(*args: str) -> str:
    data = subprocess.check_output(args, stderr=subprocess.PIPE, timeout=8)
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16").strip()
    if b"\0" in data:
        return data.decode("utf-16-le").strip()
    return data.decode("utf-8").strip()


def _windows_tool(name: str) -> str:
    return shutil.which(name) or str(Path("/mnt/c/Windows/System32") / name)


def _windows_path(value: str) -> PureWindowsPath:
    return PureWindowsPath(value.removeprefix("\\\\?\\"))


def _linux_path(path: PureWindowsPath) -> Path:
    if not path.is_absolute() or not _DRIVE.fullmatch(path.drive):
        raise ValueError(f"Not a local Windows drive path: {path}")
    return Path(_output("wslpath", "-u", str(path)))


def _registry_entries(output: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    values: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("HKEY_CURRENT_USER\\"):
            values = {}
            entries[line] = values
        else:
            match = re.match(
                r"^\s+(DistributionName|BasePath|VhdFileName)\s+REG_(?:EXPAND_)?SZ\s+(.+)$",
                line,
            )
            if match:
                values[match[1]] = match[2].strip()
    return {
        values["DistributionName"].casefold(): values
        for values in entries.values()
        if "DistributionName" in values and "BasePath" in values
    }


def _unknown(reason: str) -> Storage:
    return Storage(None, unavailable_reason=reason)


def _on_windows_drive(path: Path, windows_path: PureWindowsPath) -> bool:
    # wslpath alone can translate a missing mount to a plain directory on ext4.
    mount = _linux_path(PureWindowsPath(f"{windows_path.drive}\\"))
    return (
        mount.is_mount()
        and path.is_relative_to(mount)
        and path.stat().st_dev == mount.stat().st_dev
    )


def _vhd_storage(candidates: list[Path], windows_path: PureWindowsPath) -> Storage:
    disks = [
        path
        for path in dict.fromkeys(candidates)
        if path.is_file() and not path.is_symlink()
    ]
    if len(disks) != 1 or not _on_windows_drive(disks[0], windows_path):
        raise ValueError(f"Cannot verify a unique VHDX on {windows_path.drive}")
    return WindowsBackingStorage(disks[0], windows_path)


class WindowsBackingStorage(Storage):
    def __init__(self, path: Path, location: PureWindowsPath) -> None:
        super().__init__(path, docker=True)
        self.windows_drive = location.drive.upper()
        self.mount = _linux_path(PureWindowsPath(f"{self.windows_drive}\\"))

    def measure(self) -> tuple[Device, int] | None:
        result = self._local()
        if result is None:
            return None
        try:
            if self.mount.is_mount() and self.mount.stat().st_dev == result[0]:
                return result
        except OSError:
            pass
        self.measurement_error = (
            f"Windows drive {self.windows_drive} is no longer mounted at {self.mount}"
        )
        return None


class DockerWslStorage(Storage):
    def __init__(self, root: Path) -> None:
        super().__init__(root, docker=True)

    def authorize_probe(self) -> None:
        # Linux sudo in the CLI distro cannot elevate a Windows/WSL probe.
        return

    def measure(self) -> tuple[Device, int] | None:
        # Verify the daemon namespace's actual data root is backed by the data
        # disk, not the distro root or an unrelated filesystem.
        script = (
            'set -eu; root=$1; set -- $(pidof dockerd); [ "$#" -eq 1 ]; '
            'path=/proc/$1/root$root; dev=$(stat -c %d "$path"); '
            '[ "$dev" = "$(stat -c %d /mnt/docker-desktop-disk)" ]; '
            "free=$(df -Pk \"$path\" | awk 'NR == 2 { print $4 }'); "
            'printf "%s %s\\n" "$dev" "$free"'
        )
        try:
            wsl = _windows_tool("wsl.exe")
            running = (
                _output(wsl, "--list", "--running", "--quiet").casefold().splitlines()
            )
            if _DOCKER_DISTRO not in {name.strip() for name in running}:
                raise ValueError("Docker Desktop's WSL distribution is not running")
            device, free = map(
                int,
                _output(
                    wsl,
                    "--distribution",
                    _DOCKER_DISTRO,
                    "--cd",
                    "/",
                    "--exec",
                    "sh",
                    "-c",
                    script,
                    "probe",
                    str(self.path),
                ).split(),
            )
            if device < 0 or free < 0:
                raise ValueError("Invalid Docker Desktop filesystem counters")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.measurement_error = (
                f"Docker Desktop WSL data filesystem could not be verified: {exc}"
            )
            return None
        self.measurement_error = None
        return f"docker-wsl:{device}", free * 1024


class WslStorageProbe:
    def __init__(self) -> None:
        self._registry: dict[str, dict[str, str]] | None = None

    @classmethod
    def detect(cls) -> WslStorageProbe | None:
        try:
            release = (
                Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
            )
        except OSError:
            return None
        return (
            cls()
            if "wsl2" in release or ("microsoft" in release and "wsl" in release)
            else None
        )

    def _entry(self, distro: str) -> dict[str, str]:
        if self._registry is None:
            self._registry = _registry_entries(
                _output(_windows_tool("reg.exe"), "query", _LXSS, "/s")
            )
        if distro.casefold() not in self._registry:
            raise ValueError(f"No WSL registry entry for {distro!r}")
        return self._registry[distro.casefold()]

    def guest_host(self, path: Path) -> Storage | None:
        """Return an additional Windows budget, or none for direct Windows I/O."""
        try:
            ancestor = path.resolve()
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if ancestor.stat().st_dev != Path("/").stat().st_dev:
                translated = _windows_path(_output("wslpath", "-w", str(ancestor)))
                if _DRIVE.fullmatch(translated.drive) and _on_windows_drive(
                    ancestor, translated
                ):
                    return None
                raise ValueError(
                    f"Cannot map the WSL filesystem containing {path} to a Windows drive"
                )
            distro = os.environ.get("WSL_DISTRO_NAME")
            if not distro:
                unc = PureWindowsPath(_output("wslpath", "-w", "/"))
                match = re.fullmatch(
                    r"\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)", unc.drive, re.IGNORECASE
                )
                if not match:
                    raise ValueError("Cannot identify the current WSL distribution")
                distro = match[1]
            entry = self._entry(distro)
            base = _windows_path(entry["BasePath"])
            directory = _linux_path(base)
            name = entry.get("VhdFileName")
            if name is not None and (
                PureWindowsPath(name).name != name or not name.lower().endswith(".vhdx")
            ):
                raise ValueError("Unsupported WSL VhdFileName")
            disks = [directory / name] if name else list(directory.glob("*.vhdx"))
            return _vhd_storage(disks, base)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return _unknown(f"WSL backing storage unavailable: {exc}")

    def docker_host(self) -> Storage:
        try:
            main = _windows_path(self._entry(_DOCKER_DISTRO)["BasePath"])
            appdata = _windows_path(
                _output(_windows_tool("cmd.exe"), "/d", "/c", "echo %APPDATA%")
            )
            settings_dir = _linux_path(appdata / "Docker")
            for filename in ("settings-store.json", "settings.json"):
                settings = settings_dir / filename
                if settings.is_file():
                    data = json.loads(settings.read_text(encoding="utf-8-sig"))
                    break
            else:
                raise ValueError("Docker Desktop settings are unavailable")
            if not isinstance(data, dict):
                raise TypeError("Docker Desktop settings are not a JSON object")
            locations: set[str] = set()
            for key, value in data.items():
                if key.casefold() not in {
                    "datafolder",
                    "diskimagelocation",
                    "wsldatadir",
                }:
                    continue
                location_value = (
                    value.get("value") if isinstance(value, dict) else value
                )
                if isinstance(location_value, str) and location_value.strip():
                    locations.add(location_value.strip())
                elif (
                    location_value not in (None, "")
                    or isinstance(value, dict)
                    and "value" not in value
                ):
                    raise ValueError(
                        f"Unsupported Docker Desktop location setting {key!r}"
                    )
            if len(locations) > 1:
                raise ValueError(
                    "Docker Desktop reports conflicting data disk locations"
                )
            if locations:
                location = _windows_path(locations.pop())
                path = _linux_path(location)
                disks = (
                    [path]
                    if path.suffix.lower() == ".vhdx"
                    else [
                        path / "docker_data.vhdx",
                        path / "disk" / "docker_data.vhdx",
                        path / "wsl" / "disk" / "docker_data.vhdx",
                    ]
                )
            else:
                if main.name.casefold() != "main":
                    raise ValueError("Unsupported Docker Desktop WSL directory layout")
                location = main.parent / "disk"
                disks = [_linux_path(location) / "docker_data.vhdx"]
            return _vhd_storage(disks, location)
        except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            return _unknown(
                f"Docker Desktop Windows backing storage unavailable: {exc}"
            )
