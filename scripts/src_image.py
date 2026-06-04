from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence

DEFAULT_PROJECT_NAME = "agent-mgmt-src"
DEFAULT_IMAGE_REPOSITORY = "hieucien/agent-mgmt-src"
DEFAULT_DOCKERFILE = Path("docker/Dockerfile.src")
ZIP_ROOT = Path(".")


@dataclass(frozen=True)
class CommandResult:
    stdout: str = ""


@dataclass(frozen=True)
class SourceNames:
    project_name: str
    timestamp: str

    @property
    def zip_name(self) -> str:
        return f"{self.project_name}.{self.timestamp}.zip"

    @property
    def output_dir(self) -> str:
        return f"{self.project_name}-{self.timestamp}"


class RunCommand(Protocol):
    def __call__(
        self, command: list[str], *, cwd: Path | None = None
    ) -> CommandResult: ...


def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_command(command: list[str], *, cwd: Path | None = None) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    return CommandResult(stdout=completed.stdout)


def run_best_effort(
    command: list[str], *, cwd: Path | None = None, run: RunCommand = run_command
) -> None:
    try:
        run(command, cwd=cwd)
    except subprocess.CalledProcessError as error:
        print(
            f"cleanup command failed with exit code {error.returncode}: {' '.join(command)}"
        )


def parse_image_tag(image: str) -> str:
    image_name = image.rsplit("/", maxsplit=1)[-1]
    if ":" not in image_name:
        raise ValueError(
            "source image must include a tag, for example hieucien/agent-mgmt-src:2026-06-03_17-22-20"
        )
    return image_name.rsplit(":", maxsplit=1)[1]


def ensure_output_dir_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")


def list_git_tracked_files(repo_root: Path, run: RunCommand) -> list[Path]:
    result = run(["git", "ls-files", "-z"], cwd=repo_root)
    paths = [Path(item) for item in result.stdout.split("\0") if item]
    return [path for path in paths if (repo_root / path).is_file()]


def list_gitignored_paths(repo_root: Path, paths: Sequence[Path]) -> set[Path]:
    if not paths:
        return set()
    payload = "\0".join(path.as_posix() for path in paths) + "\0"
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=repo_root,
        input=payload,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            completed.returncode,
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return {Path(item) for item in completed.stdout.split("\0") if item}


def add_file_to_zip(
    zip_file: zipfile.ZipFile, source: Path, archive_name: Path
) -> None:
    zip_file.write(source, archive_name.as_posix())


def add_git_dir(zip_file: zipfile.ZipFile, repo_root: Path) -> None:
    git_path = repo_root / ".git"
    if git_path.is_file():
        add_file_to_zip(zip_file, git_path, Path(".git"))
        return
    for path in git_path.rglob("*"):
        if path.is_file():
            add_file_to_zip(zip_file, path, path.relative_to(repo_root))


def create_source_archive(
    repo_root: Path, archive_path: Path, run: RunCommand = run_command
) -> None:
    tracked_files = list_git_tracked_files(repo_root, run)
    ignored_paths = list_gitignored_paths(repo_root, tracked_files)
    included_files = [path for path in tracked_files if path not in ignored_paths]

    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_file:
        for path in included_files:
            add_file_to_zip(zip_file, repo_root / path, ZIP_ROOT / path)
        add_git_dir(zip_file, repo_root)


def push_source_image(
    *,
    repo_root: Path,
    project_name: str,
    image_repository: str,
    dockerfile: Path,
    timestamp: str | None = None,
    run: RunCommand = run_command,
) -> str:
    current_timestamp = timestamp or make_timestamp()
    names = SourceNames(project_name=project_name, timestamp=current_timestamp)
    archive_path = repo_root / names.zip_name
    image = f"{image_repository}:{current_timestamp}"

    try:
        create_source_archive(repo_root, archive_path, run)
        run(
            [
                "docker",
                "buildx",
                "build",
                "--load",
                "-f",
                dockerfile.as_posix(),
                "--build-arg",
                f"SRC_ZIP={names.zip_name}",
                "-t",
                image,
                ".",
            ],
            cwd=repo_root,
        )
    finally:
        archive_path.unlink(missing_ok=True)

    try:
        run(["docker", "push", image], cwd=repo_root)
    finally:
        run_best_effort(["docker", "image", "rm", image], cwd=repo_root, run=run)
    return image


def pull_source_image(
    *,
    dest_dir: Path,
    project_name: str,
    image: str,
    run: RunCommand = run_command,
) -> Path:
    timestamp = parse_image_tag(image)
    names = SourceNames(project_name=project_name, timestamp=timestamp)
    archive_path = dest_dir / names.zip_name
    output_dir = dest_dir / names.output_dir
    ensure_output_dir_available(output_dir)

    container_id: str | None = None
    try:
        run(["docker", "pull", image], cwd=dest_dir)
        container_id = run(["docker", "create", image], cwd=dest_dir).stdout.strip()
        if not container_id:
            raise RuntimeError(
                f"docker create did not return a container id for image: {image}"
            )
        run(
            [
                "docker",
                "cp",
                f"{container_id}:/src/{names.zip_name}",
                str(archive_path),
            ],
            cwd=dest_dir,
        )
        shutil.unpack_archive(archive_path, output_dir)
    finally:
        archive_path.unlink(missing_ok=True)
        if container_id:
            run_best_effort(["docker", "rm", container_id], cwd=dest_dir, run=run)
        run_best_effort(["docker", "image", "rm", image], cwd=dest_dir, run=run)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package source into a Docker image, or restore it from one."
    )
    parser.add_argument(
        "--project-name",
        default=os.environ.get("SRC_PROJECT_NAME", DEFAULT_PROJECT_NAME),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    push_parser = subparsers.add_parser("push")
    push_parser.add_argument(
        "--image-repository",
        default=os.environ.get("SRC_IMAGE_REPOSITORY", DEFAULT_IMAGE_REPOSITORY),
    )
    push_parser.add_argument(
        "--dockerfile",
        type=Path,
        default=Path(os.environ.get("SRC_DOCKERFILE", DEFAULT_DOCKERFILE.as_posix())),
    )

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument(
        "--image",
        default=os.environ.get("SRC_IMAGE"),
        required=os.environ.get("SRC_IMAGE") is None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path.cwd()
    if args.command == "push":
        image = push_source_image(
            repo_root=repo_root,
            project_name=args.project_name,
            image_repository=args.image_repository,
            dockerfile=args.dockerfile,
        )
        print(f"pushed source image: {image}")
        return 0
    if args.command == "pull":
        output_dir = pull_source_image(
            dest_dir=repo_root,
            project_name=args.project_name,
            image=args.image,
        )
        print(f"restored source to: {output_dir}")
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
