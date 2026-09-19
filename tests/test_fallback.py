from __future__ import annotations

import builtins
import importlib
import io
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch


@contextmanager
def blocked_rich():
    import src.console as console_mod
    import src.progress as progress_mod

    saved_modules = dict(sys_modules_snapshot())
    real_import = builtins.__import__

    def blocker(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    for mod in list(__import__("sys").modules):
        if mod == "rich" or mod.startswith("rich."):
            del __import__("sys").modules[mod]
    builtins.__import__ = blocker
    try:
        importlib.reload(console_mod)
        importlib.reload(progress_mod)
        yield console_mod, progress_mod
    finally:
        builtins.__import__ = real_import
        for mod in [m for m in list(__import__("sys").modules) if m == "rich" or m.startswith("rich.")]:
            del __import__("sys").modules[mod]
        for name, mod in saved_modules.items():
            if name == "rich" or name.startswith("rich."):
                __import__("sys").modules[name] = mod
        importlib.reload(console_mod)
        importlib.reload(progress_mod)


def sys_modules_snapshot():
    import sys

    return {m: sys.modules[m] for m in list(sys.modules) if m == "rich" or m.startswith("rich.")}


class FallbackTests(unittest.TestCase):
    def test_console_fallback_plain_output(self) -> None:
        with blocked_rich() as (console_mod, _):
            self.assertFalse(console_mod.RICH_AVAILABLE)
            console = console_mod.Console()
            buf = io.StringIO()
            with redirect_stdout(buf):
                console.print("[red]hello[/red] world")
            self.assertEqual(buf.getvalue().strip(), "hello world")
            table = console_mod.Table(title="T")
            table.add_column("A")
            table.add_row("x")
            self.assertIn("x", str(table))
            with patch("builtins.input", return_value=""):
                self.assertEqual(console_mod.Prompt.ask("Q", default="d"), "d")
            with patch("builtins.input", return_value="y"):
                self.assertTrue(console_mod.Confirm.ask("Q?", default=False))
            with patch("builtins.input", return_value=""):
                self.assertEqual(console_mod.IntPrompt.ask("N", default=3), 3)

    def test_progress_fallback_api(self) -> None:
        with blocked_rich() as (_, progress_mod):
            self.assertEqual(progress_mod.format_elapsed(1.2), "1.2s")
            with progress_mod.Progress() as progress:
                task = progress.add_task("job", total=10)
                progress.update(task, advance=4)
                progress.remove_task(task)
            with patch.dict(os.environ, {"NO_PROGRESS": "1"}):
                self.assertFalse(progress_mod.progress_enabled())

    def test_hf_head_failure_still_uses_cache(self) -> None:
        from src import huggingface as hf

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(cache)}, clear=False),
                patch.object(hf, "_hf_head_meta", return_value=None),
                patch.object(hf, "_hf_stream_body", side_effect=lambda url, fn, hd, target, off=0: target.write_bytes(b"data")),
            ):
                dest = hf.download_model_file("Org/Model", "a.bin", "main", Path(tmp) / "raw")
            self.assertEqual(dest.read_bytes(), b"data")
            self.assertEqual((cache / "Org_Model" / "main" / "a.bin").read_bytes(), b"data")

    def test_hf_empty_download_raises(self) -> None:
        from src import huggingface as hf

        class FakeHead:
            headers: ClassVar[dict[str, str]] = {"ETag": '"e1"', "Content-Length": "6"}

            def read(self) -> bytes:
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        class FakeEmpty:
            status = 200

            def read(self, size: int = -1) -> bytes:
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

        def fake_urlopen(request: object, timeout: int = 60) -> object:
            if getattr(request, "get_method", lambda: "GET")() == "HEAD":
                return FakeHead()
            return FakeEmpty()

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with (
                patch.object(hf, "_HF_AVAILABLE", False),
                patch.dict(os.environ, {"DOWNLOAD_LLM_CACHE": str(cache)}, clear=False),
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
                self.assertRaises(RuntimeError),
            ):
                hf.download_model_file("Org/Model", "a.bin", "main", Path(tmp) / "raw")


if __name__ == "__main__":
    unittest.main()
