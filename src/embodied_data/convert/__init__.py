from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from rich.console import Console

from embodied_data._emit import emit_error, emit_json, get_console
from embodied_data._state import state

console = get_console()

_SUPPORTED = {
    ("agibot", "lerobot-v3"),
    ("lerobot-v3", "agibot"),
}


@contextmanager
def _silence_for_json() -> Iterator[None]:
    """In --json mode, swap A's module-level rich Console for a quiet one and
    redirect stdout/stderr so the progress bar / batch ``done:`` prints don't
    leak into the JSON stream. Restored on exit.
    """
    if not state.json_output:
        yield
        return

    from embodied_data.convert import agibot_to_lerobot as _ab

    saved = _ab.console
    quiet = Console(file=io.StringIO(), force_terminal=False, no_color=True, quiet=True)
    _ab.console = quiet
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            yield
    finally:
        _ab.console = saved


def _read_dst_totals(dst: Path) -> tuple[int, int]:
    """Return (total_episodes, total_frames) by reading the freshly written info.json."""
    info_path = dst / "meta" / "info.json"
    if not info_path.is_file():
        return 0, 0
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0
    return int(info.get("total_episodes") or 0), int(info.get("total_frames") or 0)


def _read_agibot_dst_totals(dst: Path) -> tuple[int, int]:
    """Return (episodes, frames) for an AgiBot-shape output (dst/meta_info/**/proprio_states.h5)."""
    import h5py

    h5_paths = list((dst / "meta_info").rglob("proprio_states.h5")) if dst.exists() else []
    total_frames = 0
    for p in h5_paths:
        try:
            with h5py.File(p, "r") as f:
                total_frames += int(f["state/joint/position"].shape[0])
        except (OSError, KeyError):
            continue
    return len(h5_paths), total_frames


def run_convert(
    *,
    src: Path,
    dst: Path,
    from_format: str,
    to_format: str,
    max_episodes: int | None = None,
    resume: bool = False,
    workers: int = 1,
) -> None:
    pair = (from_format, to_format)
    if pair not in _SUPPORTED:
        emit_error(
            f"unsupported conversion: {from_format} -> {to_format}",
            suggestion=f"supported pairs in v0.0.1: {sorted(_SUPPORTED)}",
            exit_code=2,
        )

    if pair == ("lerobot-v3", "agibot"):
        from embodied_data.convert.lerobot_to_agibot import convert_lerobot_v3_to_agibot

        if max_episodes is not None or resume or workers > 1:
            emit_error(
                "lerobot-v3 -> agibot is single-episode in v0.1; "
                "--max-episodes / --resume / --workers are not supported for this pair",
                suggestion="run without those flags; multi-episode reverse is v0.2",
                exit_code=2,
            )

        t0 = time.monotonic()
        try:
            with _silence_for_json():
                convert_lerobot_v3_to_agibot(src=src, dst=dst)
        except FileNotFoundError as exc:
            emit_error(
                str(exc),
                suggestion=(
                    "expected a LeRobot v3 dataset directory containing meta/info.json, "
                    "meta/tasks.parquet, data/.../*.parquet, and videos/.../*.mp4"
                ),
                exit_code=2,
            )
        except NotImplementedError as exc:
            emit_error(
                str(exc),
                suggestion="v0.1 reverse only supports single-episode v3 datasets at fps=30",
                exit_code=2,
            )
        duration = time.monotonic() - t0

        if state.json_output:
            episodes_written, frames_written = _read_agibot_dst_totals(dst)
            emit_json(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "format_pair": [from_format, to_format],
                    "episodes_written": episodes_written,
                    "frames_written": frames_written,
                    "duration_seconds": round(duration, 3),
                    "warnings": [],
                }
            )
        return

    if pair == ("agibot", "lerobot-v3"):
        from embodied_data.convert.agibot_to_lerobot import (
            convert_agibot_batch,
            convert_agibot_to_lerobot_v3,
            is_batch_src,
        )

        # Batch mode triggers when --max-episodes is set OR src lacks proprio_states.h5
        # at its top level (so it must be a parent directory of episodes).
        batch = max_episodes is not None or resume or workers > 1 or is_batch_src(src)

        if resume and not (dst / "meta" / "extra" / "uuid_map.parquet").is_file():
            uuid_map = dst / "meta" / "extra" / "uuid_map.parquet"
            emit_error(
                f"--resume requested but no resume state found at {uuid_map}",
                suggestion="run without --resume for the first conversion",
                exit_code=2,
            )

        t0 = time.monotonic()
        try:
            with _silence_for_json():
                if batch:
                    convert_agibot_batch(
                        src=src,
                        dst=dst,
                        max_episodes=max_episodes,
                        resume=resume,
                        workers=workers,
                    )
                else:
                    convert_agibot_to_lerobot_v3(src=src, dst=dst)
        except FileNotFoundError as exc:
            emit_error(
                str(exc),
                suggestion=(
                    "AgiBot expects parallel meta_info/<task>/<uuid>/ and "
                    "observations/<task>/<uuid>/ trees. Check that both exist under your "
                    "dataset root."
                ),
                exit_code=2,
            )
        except ValueError as exc:
            emit_error(
                str(exc),
                suggestion=(
                    "verify --max-episodes is positive and that the source tree contains "
                    "discoverable episodes (proprio_states.h5 + task_info.json)"
                ),
                exit_code=2,
            )
        duration = time.monotonic() - t0

        if state.json_output:
            episodes_written, frames_written = _read_dst_totals(dst)
            emit_json(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "format_pair": [from_format, to_format],
                    "episodes_written": episodes_written,
                    "frames_written": frames_written,
                    "duration_seconds": round(duration, 3),
                    "warnings": [],
                }
            )
