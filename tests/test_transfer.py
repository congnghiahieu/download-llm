from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import parse_args
from src.console import Console
from src.interactive import prompt_transfer
from src.transfer import (
    choose_destination,
    destination_conflicts,
    run_transfer,
    ssh_connection,
    ssh_options,
    transfer_one,
    validate_sources,
)


class TransferTests(unittest.TestCase):
    def args(self, *extra: str):
        return parse_args(["--transfer-models", "--ssh-host", "server",
                           "--destination-path", "/weights", *extra])

    def test_cli(self):
        args = self.args("--model-name", "a,b", "--model-name", "c",
                         "--ssh-option", "Port=2222")
        self.assertEqual(args.model_names, ["a", "b", "c"])
        self.assertIsNone(args.max_workers)
        self.assertEqual(args.existing_destination, "ask")
        for extra in (["--phases", "restore_llm"], ["--list-models"],
                      ["--max-workers", "0"]):
            with self.assertRaises(SystemExit):
                self.args(*extra)
        with self.assertRaises(SystemExit):
            parse_args(["--list-models", "--dry-run"])

    def test_sources_require_done_and_matching_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            restored = model / "extracted/restored"
            restored.mkdir(parents=True)
            (restored / "weights").write_bytes(b"abc")
            state = {"raw": [{"filename": "weights", "size_bytes": 3}],
                     "restore": [{"raw_filename": "weights", "restored": True}]}
            (model / "state.json").write_text(json.dumps(state))
            args = self.args("--model-weights-dir", tmp, "--model-name", "model")
            self.assertEqual(validate_sources(args), [("model", restored)])
            (restored / "weights").write_bytes(b"bad size")
            with self.assertRaisesRegex(ValueError, "invalid restored file"):
                validate_sources(args)
            state["restore"] = []
            (model / "state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "restore status"):
                validate_sources(args)

    def test_ssh_options_and_cleanup_on_error(self):
        self.assertEqual(ssh_options("alias", ["Port=22"]), ["-o", "Port=22"])
        for host, options in [("-bad", []), ("alias", ["ControlPath=/tmp/x"])]:
            with self.assertRaises(ValueError):
                ssh_options(host, options)
        with patch("src.transfer.subprocess.run") as run:
            with self.assertRaises(RuntimeError), ssh_connection("alias", []):
                raise RuntimeError("test")
            self.assertIn("-M", run.call_args_list[0].args[0])
            self.assertIn("exit", run.call_args_list[-1].args[0])

    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH client required")
    def test_openssh_parses_master_and_transfer_sessions_with_proxyjump(self):
        with (
            patch("src.transfer.subprocess.run") as run,
            ssh_connection("target", ssh_options("target", ["ProxyJump=jump"])) as transport,
        ):
            master = run.call_args.args[0]

        for args, session_type, control_master, fork in (
            (master, "none", "true", "yes"),
            ([*transport, "target", "true"], "default", "false", "no"),
        ):
            with self.subTest(session_type=session_type):
                config = subprocess.run(
                    [args[0], "-G", "-F", "/dev/null", *args[1:]],
                    check=True, capture_output=True, text=True,
                ).stdout.splitlines()
                self.assertIn(f"sessiontype {session_type}", config)
                self.assertIn(f"controlmaster {control_master}", config)
                self.assertIn(f"forkafterauthentication {fork}", config)
                self.assertIn("proxyjump jump", config)

    @patch("src.transfer.destination_conflicts", return_value=["model"])
    def test_existing_destination_noninteractive(self, conflicts):
        with patch("src.transfer.sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "Confirmation"):
                choose_destination(self.args(), ["ssh"], ["model"])
            self.assertEqual(choose_destination(
                self.args("--existing-destination", "merge"), ["ssh"], ["model"]), "/weights")

    def test_new_destination_rechecked(self):
        with patch("src.transfer.destination_conflicts", side_effect=[["m"], []]) as check:
            with patch("src.transfer.sys.stdin.isatty", return_value=True), patch(
                "src.transfer.Prompt.ask", side_effect=["1", "/new path"]
            ):
                self.assertEqual(choose_destination(self.args(), ["ssh"], ["m"]), "/new path")
            self.assertEqual(check.call_args.args[2], "/new path")

    def test_conflicts_check_model_folder_and_quote_shell(self):
        with patch("src.transfer.remote", return_value=subprocess.CompletedProcess([], 0, "")) as remote:
            self.assertEqual(destination_conflicts(["ssh"], "host", "/a b", ["m"]), [])
            self.assertIn("'/a b/m'", remote.call_args.args[2])

    def test_rsync_command_and_failure(self):
        with patch("src.transfer.remote"), patch("src.transfer.subprocess.run") as run:
            self.assertTrue(transfer_one("m", Path("/source"), "/a b", "alias",
                                         ["ssh", "-o", "Port=22"], "rsync", False))
            command = run.call_args.args[0]
            self.assertEqual(command[-2:], ["/source/", "alias:/a b/m/"])
            self.assertIn("--partial", command)
            self.assertNotIn("--delete", command)
            run.side_effect = subprocess.CalledProcessError(1, "rsync")
            self.assertFalse(transfer_one("m", Path("/source"), "/dst", "host", ["ssh"], "rsync", False))

    def test_scp_dry_run_never_writes(self):
        with patch("src.transfer.remote") as remote, patch("src.transfer.subprocess.run") as run:
            self.assertTrue(transfer_one("m", Path("/source"), "/dst", "host", ["ssh"], "scp", True))
            remote.assert_not_called()
            run.assert_not_called()

    def test_rsync_dry_run_existing_and_absent_destination(self):
        for exists in ("exists", ""):
            with self.subTest(exists=exists), patch(
                "src.transfer.remote", return_value=subprocess.CompletedProcess([], 0, exists)
            ) as remote, patch("src.transfer.subprocess.run") as run:
                self.assertTrue(transfer_one("m", Path("/source"), "/dst", "host", ["ssh"], "rsync", True))
                self.assertNotIn("mkdir", remote.call_args.args[2])
                if exists:
                    self.assertIn("--dry-run", run.call_args.args[0])
                else:
                    run.assert_not_called()

    def test_interactive_transfer_options(self):
        with patch("src.interactive.prompt_model_names", return_value=["a", "b"]), \
                patch("src.interactive.Prompt.ask", side_effect=["alias", "/dst", "Port=2222", "", "ask"]), \
                patch("src.interactive.prompt_execution_mode", return_value=2), \
                patch("src.interactive.Confirm.ask", side_effect=[True, True]), \
                patch("src.cli.execute", return_value=0) as execute:
            self.assertEqual(prompt_transfer(Console(), "/weights"), 0)
            args = execute.call_args.args[0]
            self.assertTrue(args.transfer_models)
            self.assertTrue(args.dry_run)
            self.assertEqual(args.ssh_option, ["Port=2222"])
            self.assertEqual(args.max_workers, 2)
            self.assertEqual(args.model_names, ["a", "b"])

    def test_default_execution_is_parallel(self):
        barrier = threading.Barrier(2, timeout=5)

        def transfer(*args):
            barrier.wait()
            return True

        with patch("src.transfer.validate_sources", return_value=[("a", Path("/a")), ("b", Path("/b"))]), \
                patch("src.transfer.shutil.which", return_value="tool"), \
                patch("src.transfer.ssh_connection") as connection, \
                patch("src.transfer.remote", return_value=subprocess.CompletedProcess([], 0, "rsync")), \
                patch("src.transfer.choose_destination", return_value="/dst"), \
                patch("src.transfer.transfer_one", side_effect=transfer):
            connection.return_value.__enter__.return_value = ["ssh"]
            self.assertEqual(run_transfer(self.args()), 0)

    def test_scp_payload_includes_dotfiles_without_extra_layer(self):
        with patch("src.transfer.remote"), patch("src.transfer.subprocess.run") as run, patch(
            "src.transfer.Path.iterdir", return_value=iter([Path("/source/.hidden")])
        ):
            self.assertTrue(transfer_one("m", Path("/source"), "/a b", "host", ["ssh"], "scp", False))
            self.assertEqual(run.call_args.args[0][-2:], ["/source/.hidden", "host:'/a b/m/'"])

    def test_fallback_and_other_models_continue(self):
        with patch("src.transfer.validate_sources", return_value=[("a", Path("/a")), ("b", Path("/b"))]), \
                patch("src.transfer.shutil.which", side_effect=lambda cmd: None if cmd == "rsync" else cmd), \
                patch("src.transfer.ssh_connection") as connection, \
                patch("src.transfer.remote", return_value=subprocess.CompletedProcess([], 0, "rsync")), \
                patch("src.transfer.choose_destination", return_value="/dst"), \
                patch("src.transfer.transfer_one", side_effect=[False, True]) as transfer:
            connection.return_value.__enter__.return_value = ["ssh"]
            self.assertEqual(run_transfer(self.args("--max-workers", "1")), 1)
            self.assertEqual(transfer.call_count, 2)
            self.assertEqual(transfer.call_args.args[-2], "scp")

    @unittest.skipUnless(shutil.which("rsync") and shutil.which("scp"), "rsync/scp required")
    def test_real_transfer_protocols_with_local_shell_transport(self):
        # Exercise real rsync and legacy SCP protocols, replacing only the SSH
        # network/auth layer with a local shell. No SSH server required.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = root / "ssh-local"
            transport.write_text(
                '#!/bin/sh\n'
                'while [ "$#" -gt 0 ]; do\n'
                'case "$1" in\n'
                '  -o|-l|-p|-i) shift 2 ;;\n'
                '  -*) shift ;;\n'
                '  *) shift; break ;;\n'
                'esac\ndone\n'
                'exec /bin/sh -c "$*"\n'
            )
            transport.chmod(0o700)
            source = root / "source"
            (source / "assets").mkdir(parents=True)
            (source / "assets/weights").write_bytes(b"model data")
            (source / ".hidden").write_text("hidden")
            (source / "empty").mkdir()
            for backend in ("rsync", "scp"):
                destination = root / f"{backend} destination's space"
                with self.subTest(backend=backend):
                    self.assertTrue(transfer_one("model", source, str(destination), "local",
                                                 [str(transport)], backend, False))
                    model = destination / "model"
                    self.assertEqual((model / "assets/weights").read_bytes(), b"model data")
                    self.assertEqual((model / ".hidden").read_text(), "hidden")
                    self.assertTrue((model / "empty").is_dir())
                    (model / "extra").write_text("keep")
                    (source / "assets/weights").write_bytes(b"updated model data")
                    self.assertTrue(transfer_one("model", source, str(destination), "local",
                                                 [str(transport)], backend, False))
                    self.assertEqual((model / "assets/weights").read_bytes(), b"updated model data")
                    self.assertTrue((model / "extra").exists())
                    (source / "assets/weights").write_bytes(b"model data")
