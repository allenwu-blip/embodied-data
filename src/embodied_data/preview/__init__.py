from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from embodied_data.preview.agibot import collect_agibot_stats
from embodied_data.preview.lerobot_v3 import collect_lerobot_v3_stats

console = Console()


def detect_format(path: Path) -> str | None:
    """Return 'lerobot-v3' | 'agibot' | None.

    Self-contained to avoid a circular import with validate/ during Sprint 1.
    """
    if not path.exists() or not path.is_dir():
        return None
    if (path / "meta" / "info.json").is_file():
        return "lerobot-v3"
    if (path / "proprio_states.h5").is_file():
        return "agibot"
    # convert_to_lerobot.py lives at <root>/scripts/, so an episode dir at
    # <root>/meta_info/<task>/<uuid>/ is 3 levels deep relative to root.
    for up in (path, *path.parents[:4]):
        if (up / "scripts" / "convert_to_lerobot.py").is_file():
            return "agibot"
    return None


def run_preview(*, path: Path, n: int) -> None:
    """Print a stats table for the first N episodes of an AgiBot or LeRobot v3 dataset."""
    if n <= 0:
        n = 10
    fmt = detect_format(path)
    if fmt is None:
        console.print(
            f"[red]error:[/red] cannot read {path} or unknown format "
            "(expected meta/info.json for lerobot-v3, or proprio_states.h5 for agibot).",
        )
        raise typer.Exit(code=2)

    try:
        if fmt == "lerobot-v3":
            stats, header = collect_lerobot_v3_stats(path, n)
        else:
            stats, header = collect_agibot_stats(path, n)
    except (OSError, ValueError, KeyError) as exc:
        console.print(f"[red]error:[/red] failed to read {fmt} dataset at {path}: {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"Preview: {path}  (format: {fmt})")
    if header:
        console.print(header)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for field, value in stats:
        table.add_row(field, str(value))
    console.print(table)
