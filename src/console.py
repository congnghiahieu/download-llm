from __future__ import annotations

import getpass
import re
import sys
from typing import Any

try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.prompt import Confirm as _RichConfirm
    from rich.prompt import IntPrompt as _RichIntPrompt
    from rich.prompt import Prompt as _RichPrompt
    from rich.table import Table as _RichTable

    Console: Any = _RichConsole
    Panel: Any = _RichPanel
    Confirm: Any = _RichConfirm
    IntPrompt: Any = _RichIntPrompt
    Prompt: Any = _RichPrompt
    Table: Any = _RichTable
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    def strip_markup(text: str) -> str:
        return re.sub(r"\[.*?\]", "", text)

    def render_plain_table(title: str, columns: list[str], rows: list[list[str]]) -> str:
        widths = [len(header) for header in columns]
        for row in rows:
            for index, cell in enumerate(row):
                if index < len(widths):
                    widths[index] = max(widths[index], len(cell))
        lines = [title] if title else []
        lines.append(
            " | ".join(header.ljust(widths[index]) for index, header in enumerate(columns))
        )
        lines.append("-+-".join("-" * width for width in widths))
        for row in rows:
            lines.append(
                " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
            )
        return "\n".join(lines)

    class Console:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._stderr = bool(kwargs.get("stderr", False))

        def print(self, *args: object, **kwargs: object) -> None:
            text = " ".join(strip_markup(str(arg)) for arg in args)
            print(text, file=sys.stderr if self._stderr else sys.stdout, flush=True)

    class Panel:  # type: ignore[no-redef]
        def __init__(self, text: object, *args: object, **kwargs: object) -> None:
            self.text = str(text)
            self.title = str(kwargs.get("title", ""))

        def __str__(self) -> str:
            if self.title:
                return f"{self.title}: {self.text}"
            return self.text

    class Prompt:  # type: ignore[no-redef]
        @staticmethod
        def ask(
            prompt: str,
            *args: object,
            choices: list[str] | None = None,
            default: object = "",
            password: bool = False,
            **kwargs: object,
        ) -> str:
            default_str = "" if default is None else str(default)
            while True:
                if choices:
                    suffix = f" [{'/'.join(choices)}] (default: {default_str})"
                elif default_str:
                    suffix = f" (default: {default_str})"
                else:
                    suffix = ""
                reader = getpass.getpass if password else input
                entered = reader(f"{strip_markup(prompt)}{suffix}: ").strip()
                if not entered:
                    entered = default_str
                if choices and entered not in choices:
                    print(
                        f"Invalid choice: {entered}. Choose from {', '.join(choices)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                return entered

    class Confirm:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt: str, *args: object, default: bool = False, **kwargs: object) -> bool:
            hint = "Y/n" if default else "y/N"
            while True:
                entered = input(f"{strip_markup(prompt)} [{hint}]: ").strip().lower()
                if not entered:
                    return default
                if entered in ("y", "yes"):
                    return True
                if entered in ("n", "no"):
                    return False
                print("Use y or n.", file=sys.stderr, flush=True)

    class IntPrompt:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt: str, *args: object, default: int = 0, **kwargs: object) -> int:
            while True:
                entered = input(
                    f"{strip_markup(prompt)} (default: {default}): "
                ).strip()
                if not entered:
                    return int(default)
                try:
                    return int(entered)
                except ValueError:
                    print(f"Invalid number: {entered}", file=sys.stderr, flush=True)

    class Table:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.title = str(kwargs.get("title", ""))
            self.columns: list[str] = []
            self.rows: list[list[str]] = []

        def add_column(self, header: object, **kwargs: object) -> None:
            self.columns.append(str(header))

        def add_row(self, *cells: object) -> None:
            self.rows.append([str(cell) for cell in cells])

        def __str__(self) -> str:
            return render_plain_table(self.title, self.columns, self.rows)
