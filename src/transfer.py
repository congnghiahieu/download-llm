"""Transfer restored model payloads using the user's OpenSSH installation."""

from __future__ import annotations

import argparse
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from src.console import Prompt
from src.state import effective_base_dir, load_state, phase_status, resolve_model_names


def validate_sources(args: argparse.Namespace) -> list[tuple[str, Path]]:
    root = effective_base_dir(args)
    names = dict.fromkeys(resolve_model_names(args, "transfer", base_dir=root))
    sources = []
    for name in names:
        model = root / name
        state = load_state(model)
        if not phase_status(state)["restore_llm"]:
            raise ValueError(f"Model '{name}': restore status must be done")
        source = model / "extracted" / "restored"
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"Invalid restored directory: {source}")
        for item in state["raw"]:
            file = source / item["filename"]
            if (not file.resolve().is_relative_to(source.resolve())
                    or not file.is_file()
                    or file.stat().st_size != item["size_bytes"]):
                raise ValueError(f"Missing or invalid restored file: {file}")
        # SCP and rsync have different symlink semantics. Reject rather than
        # accidentally copying data outside the restored payload with SCP.
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError(f"Symlinks are not supported in restored payload: {source}")
        sources.append((name, source.resolve()))
    return sources


def ssh_options(host: str, options: list[str]) -> list[str]:
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("Invalid SSH host; use username@hostname or an SSH alias")
    result = []
    for option in options:
        key = re.split(r"[=\s]", option.strip(), maxsplit=1)[0].lower()
        if key in {"controlmaster", "controlpath", "controlpersist", "sessiontype",
                   "remotecommand", "requesttty", "forkafterauthentication"}:
            raise ValueError(f"SSH option managed by transfer: {key}")
        result.extend(["-o", option])
    return result


@contextmanager
def ssh_connection(host: str, options: list[str]) -> Iterator[list[str]]:
    with tempfile.TemporaryDirectory(prefix="download-llm-ssh-") as directory:
        common = ["ssh", "-o", f"ControlPath={directory}/socket",
                  "-o", "ControlPersist=no", "-o", "RequestTTY=no",
                  "-o", "RemoteCommand=none"]
        transport = [*common, "-o", "ControlMaster=no", "-o", "SessionType=default",
                     "-o", "ForkAfterAuthentication=no", *options]
        master = [*common, "-o", "SessionType=none", "-M", "-f", *options, host]
        try:
            # Foreground authentication inherits the terminal, including MFA and
            # host-key prompts. Only OpenSSH backgrounds itself after success.
            subprocess.run(master, check=True)
            yield transport
        finally:
            subprocess.run([*transport, "-O", "exit", host],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)


def remote(transport: list[str], host: str, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*transport, host, command], text=True,
                          stdout=subprocess.PIPE, check=True)


def normalize_destination(path: str) -> str:
    # Absolute paths avoid different tilde/relative-path interpretation between
    # legacy SCP, SFTP-backed SCP and the remote shell.
    if not path.startswith("/") or "\x00" in path or "\n" in path:
        raise ValueError("Destination must be an absolute server path (no ~)")
    return posixpath.normpath(path)


def destination_conflicts(transport: list[str], host: str, destination: str,
                          names: list[str]) -> list[str]:
    conflicts = []
    for name in names:
        path = shlex.quote(posixpath.join(destination, name))
        command = (
            f"if [ -L {path} ]; then exit 2; "
            f"elif [ -d {path} ]; then "
            f"find {path} -mindepth 1 -maxdepth 1 -print -quit; "
            f"elif [ -e {path} ]; then exit 2; fi"
        )
        if remote(transport, host, command).stdout:
            conflicts.append(name)
    return conflicts


def choose_destination(args: argparse.Namespace, transport: list[str],
                       names: list[str]) -> str:
    destination = normalize_destination(args.destination_path)
    while True:
        conflicts = destination_conflicts(transport, args.ssh_host, destination, names)
        if not conflicts or args.existing_destination == "merge":
            return destination
        print(f"Non-empty destination model folders: {', '.join(conflicts)}", flush=True)
        if not sys.stdin.isatty():
            raise ValueError("Confirmation requires a terminal; use --existing-destination merge explicitly")
        choice = Prompt.ask(
            "1: New destination (recommended), 2: Merge (overwrite matching files, keep extras), 3: Cancel",
            choices=["1", "2", "3"], default="1",
        )
        if choice == "3":
            raise ValueError("Transfer cancelled; no payload was transferred")
        if choice == "2":
            return destination
        destination = normalize_destination(Prompt.ask("New absolute destination path").strip())


def transfer_one(name: str, source: Path, destination: str, host: str,
                 transport: list[str], backend: str, dry_run: bool) -> bool:
    target = posixpath.join(destination, name)
    try:
        print(f"[{name}] {backend}: {source}/ -> {host}:{target}/", flush=True)
        if backend == "scp" and dry_run:
            print(f"[{name}] Plan only: SCP copies all files and may overwrite matches; no diff available.")
            return True
        if dry_run and not remote(
            transport, host,
            f"if [ -d {shlex.quote(target)} ]; then printf exists; fi",
        ).stdout:
            print(f"[{name}] New destination: all payload files would be copied (no remote writes).")
            return True
        if not dry_run:
            remote(transport, host, f"mkdir -p -- {shlex.quote(target)}")
        if backend == "rsync":
            command = ["rsync", "-rlpt", "--partial", "--progress", "--human-readable",
                       "--protect-args", "-e", shlex.join(transport)]
            if dry_run:
                command.extend(["--dry-run", "--itemize-changes"])
            command.extend(["--", f"{source}/", f"{host}:{target}/"])
        else:
            # Explicit legacy protocol gives consistent shell quoting on both
            # Ubuntu 22.04 and newer OpenSSH versions (SFTP is otherwise default).
            # Legacy SCP rejects a source named '.'. Enumerating immediate
            # children also includes dotfiles without shell glob expansion and
            # bounds command size for large repositories.
            for child in sorted(source.iterdir()):
                command = ["scp", "-O", "-r", "-p", "-S", transport[0], *transport[1:], "--",
                           str(child), f"{host}:{shlex.quote(target + '/')}"]
                subprocess.run(command, check=True)
            return True
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[{name}] Failed: {exc}", file=sys.stderr, flush=True)
        return False


def run_transfer(args: argparse.Namespace) -> int:
    try:
        sources = validate_sources(args)
        options = ssh_options(args.ssh_host, args.ssh_option)
        normalize_destination(args.destination_path)
        if not shutil.which("ssh"):
            raise ValueError("OpenSSH client 'ssh' is required")
        with ssh_connection(args.ssh_host, options) as transport:
            available = remote(transport, args.ssh_host,
                               "if command -v rsync >/dev/null 2>&1; then printf rsync; fi").stdout
            backend = "rsync" if shutil.which("rsync") and available == "rsync" else "scp"
            if backend == "scp":
                print("Warning: rsync missing on source or destination; using SCP. "
                      "No resume or unchanged-file skipping; interrupted files are copied again.",
                      file=sys.stderr, flush=True)
                if not shutil.which("scp"):
                    raise ValueError("SCP fallback requires local scp")
                remote(transport, args.ssh_host, "command -v scp >/dev/null")
            destination = choose_destination(args, transport, [name for name, _ in sources])
            workers = args.max_workers or len(sources)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(transfer_one, name, source, destination,
                                           args.ssh_host, transport, backend, args.dry_run)
                           for name, source in sources]
                try:
                    results = [future.result() for future in futures]
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    raise
            for (name, _), success in zip(sources, results):
                print(f"{name}: {'done' if success else 'failed'}", flush=True)
            print(f"{'Dry-run' if args.dry_run else 'Transfer'}: "
                  f"{sum(results)} succeeded, {len(results) - sum(results)} failed")
            return 0 if all(results) else 1
    except (ValueError, OSError, subprocess.CalledProcessError, EOFError) as exc:
        print(f"Transfer error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Transfer interrupted.", file=sys.stderr)
        return 130
