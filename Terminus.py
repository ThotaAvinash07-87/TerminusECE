#!/usr/bin/env python3
"""TerminusECE - Command-Driven Terminal Workspace for Electrical and Computer Engineering."""

import argparse
import sys
import os

from UI.app import TerminusApp, TerminusEngineBridge
from CORE.ipc_router import IPCRouter
from CORE.common_math import split_smart_statements


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_cli_commands(commands: str) -> None:
    bridge = TerminusEngineBridge()
    for line in split_smart_statements(commands, ";"):
        line = line.strip()
        if not line:
            continue
        try:
            res = bridge.execute_command(line)
            if res:
                # Strip textual rich markup tags for clean CLI stdout output
                clean_text = res.replace("[bold green]", "").replace("[/bold green]", "") \
                                .replace("[bold cyan]", "").replace("[/bold cyan]", "") \
                                .replace("[bold magenta]", "").replace("[/bold magenta]", "") \
                                .replace("[bold yellow]", "").replace("[/bold yellow]", "") \
                                .replace("[bold red]", "").replace("[/bold red]", "") \
                                .replace("[green]", "").replace("[/green]", "") \
                                .replace("[red]", "").replace("[/red]", "") \
                                .replace("[yellow]", "").replace("[/yellow]", "")
                print(clean_text)
        except Exception as e:
            print(f"Error executing '{line}': {e}", file=sys.stderr)


def run_script_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"Error: Script file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    run_cli_commands(content.replace("\n", ";"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TerminusECE - Terminal-based unified workspace for ECE"
    )
    parser.add_argument(
        "--cmd", "-c",
        type=str,
        help="Execute semicolon-separated commands in CLI batch mode and exit"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Execute commands from a script file and exit"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Start the background IPC synchronization server daemon"
    )

    args = parser.parse_args()

    if args.daemon:
        print("Starting TerminusECE IPC Daemon on 127.0.0.1:8765...")
        router = IPCRouter()
        router.start_background()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            router.stop()
            print("\nDaemon stopped.")
        return

    if args.cmd:
        run_cli_commands(args.cmd)
        return

    if args.file:
        run_script_file(args.file)
        return

    # Interactive TUI mode
    app = TerminusApp()
    app.run()


if __name__ == "__main__":
    main()