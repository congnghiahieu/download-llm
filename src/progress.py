from __future__ import annotations

import os
import sys
from typing import Any

try:
    from rich.progress import BarColumn as _RichBarColumn
    from rich.progress import Progress as _RichProgress
    from rich.progress import TextColumn as _RichTextColumn
    from rich.progress import TimeRemainingColumn as _RichTimeRemainingColumn

    BarColumn: Any = _RichBarColumn
    Progress: Any = _RichProgress
    TextColumn: Any = _RichTextColumn
    TimeRemainingColumn: Any = _RichTimeRemainingColumn
except ImportError:

    class _DummyColumn:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    BarColumn = _DummyColumn
    TextColumn = _DummyColumn
    TimeRemainingColumn = _DummyColumn

    class Progress:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def __enter__(self) -> Progress:  # noqa: PYI034 - fallback must match rich Progress API
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def add_task(self, *args: object, **kwargs: object) -> int:
            return 0

        def update(self, *args: object, **kwargs: object) -> None:
            return None

        def remove_task(self, *args: object, **kwargs: object) -> None:
            return None


def progress_enabled() -> bool:
    if os.environ.get("NO_PROGRESS", "").lower() in ("1", "true", "yes"):
        return False
    return sys.stderr.isatty()


def make_progress(description: str) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        transient=True,
        disable=not progress_enabled(),
    )


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{secs:.0f}s"
