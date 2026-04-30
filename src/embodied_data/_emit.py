"""JSON and error-emission helpers shared by every command.

In JSON mode all stdout goes through ``emit_json`` so callers can pipe results
into ``jq`` / ``python -m json.tool`` deterministically. In non-JSON mode we
fall back to rich.Console with the standard error+suggestion format.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console

from embodied_data._state import state

# Single console reused everywhere the CLI prints. Stays attached to stdout.
_console = Console()


def get_console() -> Console:
    return _console


def emit_json(payload: dict[str, Any]) -> None:
    """Write one JSON line to stdout (no rich formatting)."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def emit_error(message: str, *, suggestion: str | None, exit_code: int) -> None:
    """Print a structured error and raise SystemExit.

    Honors --json: in JSON mode emits ``{"error", "suggestion", "exit_code"}``.
    Uses SystemExit (not typer.Exit) so direct callers — tests invoking the
    ``run_*`` functions outside the Click context — get a stdlib exception.
    Click/Typer transparently translate SystemExit into the right exit code.
    """
    if state.json_output:
        emit_json({"error": message, "suggestion": suggestion, "exit_code": exit_code})
    else:
        _console.print(f"[red]error:[/red] {message}")
        if suggestion:
            _console.print(f"[dim]suggestion:[/dim] {suggestion}")
    raise SystemExit(exit_code)
