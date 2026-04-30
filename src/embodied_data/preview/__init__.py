from __future__ import annotations

from pathlib import Path

from rich.table import Table

from embodied_data._agibot_paths import find_proprio_h5
from embodied_data._emit import emit_error, emit_json, get_console
from embodied_data._state import state
from embodied_data.preview.agibot import collect_agibot_stats
from embodied_data.preview.lerobot_v3 import collect_lerobot_v3_stats

console = get_console()


def detect_format(path: Path) -> str | None:
    """Return 'lerobot-v3' | 'agibot' | None.

    Self-contained to avoid a circular import with validate/ during Sprint 1.
    """
    if not path.exists() or not path.is_dir():
        return None
    if (path / "meta" / "info.json").is_file():
        return "lerobot-v3"
    if find_proprio_h5(path) is not None:
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

    if not path.exists():
        emit_error(
            f"path does not exist: {path}",
            suggestion="expected a directory; pass the dataset root, not a single file",
            exit_code=2,
        )
    if not path.is_dir():
        emit_error(
            f"not a directory: {path}",
            suggestion="expected a directory; pass the dataset root, not a single file",
            exit_code=2,
        )

    fmt = detect_format(path)
    if fmt is None:
        emit_error(
            f"cannot read {path} or unknown format",
            suggestion=(
                "expected meta/info.json (lerobot-v3) or proprio_state[s]*.h5 (agibot) in path"
            ),
            exit_code=2,
        )
        return  # unreachable

    try:
        if fmt == "lerobot-v3":
            stats, header = collect_lerobot_v3_stats(path, n)
        else:
            stats, header = collect_agibot_stats(path, n)
    except (OSError, ValueError, KeyError) as exc:
        emit_error(
            f"failed to read {fmt} dataset at {path}: {exc}",
            suggestion=(
                "verify the dataset is intact; for agibot, ensure proprio_state[s]*.h5 contains "
                "/state/joint/position and a sibling observations/<task>/<uuid>/video tree exists"
            ),
            exit_code=2,
        )
        return  # unreachable

    if state.json_output:
        emit_json(
            {
                "path": str(path),
                "format": fmt,
                "stats": dict(stats),
                "header": header or None,
            }
        )
        return

    console.print(f"Preview: {path}  (format: {fmt})")
    if header:
        console.print(header)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for field, value in stats:
        table.add_row(field, str(value))
    console.print(table)
