from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from rich.console import Console
from rich.table import Table

from embodied_data._agibot_paths import PROPRIO_GLOB
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
    """Return (episodes, frames) for an AgiBot-shape output (dst/meta_info/**/proprio_state*.h5)."""
    import h5py

    h5_paths = list((dst / "meta_info").rglob(PROPRIO_GLOB)) if dst.exists() else []
    total_frames = 0
    for p in h5_paths:
        try:
            with h5py.File(p, "r") as f:
                total_frames += int(f["state/joint/position"].shape[0])
        except (OSError, KeyError):
            continue
    return len(h5_paths), total_frames


def _run_post_convert_verify(dst: Path, *, from_format: str, to_format: str) -> None:
    """Run validate on dst after a successful convert; propagate exit code.

    Skipped on the lerobot-v3 -> agibot reverse pair (no agibot-output validator
    in v0.1; that's a v0.4 candidate). Caller is responsible for skipping when
    dry_run=True (no files on disk to validate).
    """
    if (from_format, to_format) == ("lerobot-v3", "agibot"):
        if not state.json_output:
            console.print(
                "[dim]--verify skipped: lerobot-v3 -> agibot has no output-side "
                "validator yet (v0.4 candidate).[/dim]"
            )
        return

    from embodied_data.validate import run_validate

    run_validate(path=dst, fmt="auto")


def _h5_frame_count(h5_path: Path) -> int | None:
    """Return ``state/joint/position`` row count, or ``None`` on read failure.

    Cheap: opens the h5 header and reads ``shape[0]`` only. No data decode.
    """
    import h5py

    try:
        with h5py.File(h5_path, "r") as f:
            j = f.get("state/joint/position")
            if j is None or j.ndim != 2:
                return None
            return int(j.shape[0])
    except (OSError, KeyError, ValueError):
        return None


def _estimate_h264_mb(n_frames: int) -> int:
    """Rough h264 reencode size estimate at FPS=30, ~5 Mbps avg.

    5 Mbps / 30 fps = ~21 KB/frame; round up to 25 KB to stay conservative.
    """
    return max(1, int(n_frames * 25 // 1024))


def _build_dry_run_summary(
    *,
    src: Path,
    dst: Path,
    from_format: str,
    to_format: str,
    variant: str,
    max_episodes: int | None,
) -> dict:
    """Build the dry-run summary dict for an agibot -> lerobot-v3 conversion.

    Cheap-only estimation: reads h5 headers (fast) but never decodes video or
    full proprio arrays. Sets fields to ``None`` if a value would require an
    expensive read.
    """
    from embodied_data._agibot_paths import find_proprio_h5
    from embodied_data.convert.agibot_beta_to_lerobot import (
        _discover_beta_episodes,
        find_head_color_video,
        is_beta_batch_src,
    )
    from embodied_data.convert.agibot_to_lerobot import _discover_episodes, is_batch_src

    estimated_total_frames: int | None
    estimated_videos: int | None
    estimated_output_mb: int | None

    if variant == "beta":
        single_h5 = find_proprio_h5(src)
        if single_h5 is not None and not is_beta_batch_src(src):
            mode = "single-episode"
            estimated_episodes = 1
            n = _h5_frame_count(single_h5)
            estimated_total_frames = n
            head_video = find_head_color_video(src)
            estimated_videos = 1 if head_video is not None else 0
            input_mb = single_h5.stat().st_size // (1024 * 1024)
            video_mb = _estimate_h264_mb(n) if (n is not None and head_video is not None) else 0
            estimated_output_mb = input_mb + video_mb
        else:
            mode = "batch"
            episodes = _discover_beta_episodes(src)
            if max_episodes is not None:
                episodes = episodes[:max_episodes]
            estimated_episodes = len(episodes)
            total_frames = 0
            video_count = 0
            input_mb_sum = 0
            bail = False
            for ep in episodes:
                n = _h5_frame_count(ep.h5_path)
                if n is None:
                    bail = True
                    break
                total_frames += n
                if ep.head_video_src is not None:
                    video_count += 1
                input_mb_sum += ep.h5_path.stat().st_size // (1024 * 1024)
            estimated_total_frames = None if bail else total_frames
            estimated_videos = video_count
            if estimated_total_frames is not None:
                video_mb = _estimate_h264_mb(estimated_total_frames) if video_count else 0
                estimated_output_mb = input_mb_sum + video_mb
            else:
                estimated_output_mb = None
    else:  # digitalworld
        single_h5 = find_proprio_h5(src)
        if single_h5 is not None and not is_batch_src(src):
            mode = "single-episode"
            estimated_episodes = 1
            n = _h5_frame_count(single_h5)
            estimated_total_frames = n
            estimated_videos = 1  # sim layout always has head.mp4
            input_mb = single_h5.stat().st_size // (1024 * 1024)
            estimated_output_mb = input_mb + (_estimate_h264_mb(n) if n is not None else 0)
        else:
            mode = "batch"
            episodes = _discover_episodes(src)
            if max_episodes is not None:
                episodes = episodes[:max_episodes]
            estimated_episodes = len(episodes)
            total_frames = 0
            input_mb_sum = 0
            bail = False
            for ep in episodes:
                n = _h5_frame_count(ep.h5_path)
                if n is None:
                    bail = True
                    break
                total_frames += n
                input_mb_sum += ep.h5_path.stat().st_size // (1024 * 1024)
            estimated_total_frames = None if bail else total_frames
            estimated_videos = estimated_episodes
            if estimated_total_frames is not None:
                estimated_output_mb = input_mb_sum + _estimate_h264_mb(estimated_total_frames)
            else:
                estimated_output_mb = None

    return {
        "src": str(src),
        "dst": str(dst),
        "would_create_dst": not dst.exists(),
        "format_pair": [from_format, to_format],
        "variant": variant,
        "mode": mode,
        "estimated_episodes": estimated_episodes,
        "estimated_total_frames": estimated_total_frames,
        "estimated_videos": estimated_videos,
        "estimated_output_mb": estimated_output_mb,
    }


def _print_dry_run_table(summary: dict) -> None:
    """Render the dry-run summary as a Rich table (mirrors preview/style)."""
    console.print(f"[bold]dry-run:[/bold] {summary['src']} -> {summary['dst']}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("format_pair", " -> ".join(summary["format_pair"]))
    table.add_row("variant", str(summary["variant"]))
    table.add_row("mode", str(summary["mode"]))
    table.add_row("would_create_dst", str(summary["would_create_dst"]))
    table.add_row("episodes", str(summary["estimated_episodes"]))
    table.add_row(
        "estimated_total_frames",
        "n/a"
        if summary["estimated_total_frames"] is None
        else str(summary["estimated_total_frames"]),
    )
    table.add_row(
        "estimated_videos",
        "n/a" if summary["estimated_videos"] is None else str(summary["estimated_videos"]),
    )
    table.add_row(
        "estimated_output_mb",
        "n/a"
        if summary["estimated_output_mb"] is None
        else f"~{summary['estimated_output_mb']} MB",
    )
    console.print(table)
    console.print("[dim]no files written (dry-run)[/dim]")


def _dispatch_dry_run(
    *,
    src: Path,
    dst: Path,
    from_format: str,
    to_format: str,
    max_episodes: int | None,
) -> None:
    """Handle ``--dry-run`` for the agibot -> lerobot-v3 pair only.

    Other supported pairs (currently only ``lerobot-v3 -> agibot``) exit with
    code 2 + a "not supported yet" message — that's a v0.3.0 scope guard.
    """
    if (from_format, to_format) != ("agibot", "lerobot-v3"):
        emit_error(
            f"dry-run not supported for this pair yet: {from_format} -> {to_format}",
            suggestion="dry-run currently covers agibot -> lerobot-v3 only (v0.3.0)",
            exit_code=2,
        )

    from embodied_data._agibot_paths import detect_agibot_variant, schema_summary

    variant = detect_agibot_variant(src)
    if variant == "alpha":
        # Alpha schemas are equivalent to Beta — see the alpha-routing comment in
        # run_convert. Mirror that here so the dry-run summary reflects what the
        # real convert would do.
        if not state.json_output:
            console.print(
                "[yellow]note:[/yellow] Alpha layout detected; routing through "
                "the Beta converter for the dry-run summary."
            )
        variant = "beta"
    if variant == "unknown":
        emit_error(
            f"could not identify AgiBot variant at {src}",
            suggestion=(
                f"schema summary: {schema_summary(src)}. expected sim DigitalWorld "
                "(proprio_states.h5 with 34-dim joint + name attr) or real Beta "
                "(proprio_stats.h5 with 14-dim joint, no name attr)"
            ),
            exit_code=2,
        )

    summary = _build_dry_run_summary(
        src=src,
        dst=dst,
        from_format=from_format,
        to_format=to_format,
        variant=variant,
        max_episodes=max_episodes,
    )

    if state.json_output:
        emit_json(summary)
    else:
        _print_dry_run_table(summary)


def run_convert(
    *,
    src: Path,
    dst: Path,
    from_format: str,
    to_format: str,
    max_episodes: int | None = None,
    resume: bool = False,
    workers: int = 1,
    verify: bool = False,
    dry_run: bool = False,
) -> None:
    pair = (from_format, to_format)
    if pair not in _SUPPORTED:
        emit_error(
            f"unsupported conversion: {from_format} -> {to_format}",
            suggestion=f"supported pairs in v0.0.1: {sorted(_SUPPORTED)}",
            exit_code=2,
        )

    if dry_run:
        _dispatch_dry_run(
            src=src,
            dst=dst,
            from_format=from_format,
            to_format=to_format,
            max_episodes=max_episodes,
        )
        return

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
        if verify and not dry_run:
            _run_post_convert_verify(dst, from_format=from_format, to_format=to_format)
        return

    if pair == ("agibot", "lerobot-v3"):
        from embodied_data._agibot_paths import detect_agibot_variant, schema_summary

        variant = detect_agibot_variant(src)

        if variant == "alpha":
            # Alpha schemas were empirically verified equivalent to Beta on
            # 2026-04-30 (Alpha task 389/episode 656913 vs Beta task 675/episode
            # 936938 — both 14-dim joint float64, int64 ns timestamps, missing
            # state/joint.attrs["name"], identical state/{head,waist,end,robot}
            # subgroups). The Beta converter handles Alpha cleanly. Route alpha
            # → beta with a single console note so the user knows.
            if not state.json_output:
                console.print(
                    "[yellow]note:[/yellow] Alpha layout detected; schemas are equivalent "
                    "to Beta per upstream README + empirical verification "
                    "(see docs/schema/overview.md). Routing through the Beta converter."
                )
            variant = "beta"  # fall through to beta branch below

        if variant == "unknown":
            emit_error(
                f"could not identify AgiBot variant at {src}",
                suggestion=(
                    f"schema summary: {schema_summary(src)}. expected sim DigitalWorld "
                    "(proprio_states.h5 with 34-dim joint + name attr) or real Beta "
                    "(proprio_stats.h5 with 14-dim joint, no name attr)"
                ),
                exit_code=2,
            )

        if variant == "beta":
            from embodied_data._agibot_paths import find_proprio_h5
            from embodied_data.convert.agibot_beta_to_lerobot import (
                convert_agibot_beta_batch,
                convert_agibot_beta_to_lerobot_v3,
                is_beta_batch_src,
            )

            # Auto-batch when src is a Beta task root or any of the batch flags
            # are set. Single-episode otherwise.
            beta_batch = max_episodes is not None or resume or workers > 1 or is_beta_batch_src(src)
            if beta_batch and find_proprio_h5(src) is not None:
                emit_error(
                    f"--max-episodes / --resume / --workers given but {src} looks like "
                    "a single-episode dir (has proprio_stats.h5 directly inside)",
                    suggestion=(
                        "for batch mode, point at the Beta task-dataset root containing "
                        "<task>/<ep_id>/proprio_stats.h5 + task_info_<task>.json"
                    ),
                    exit_code=2,
                )

            if (
                resume
                and beta_batch
                and not (dst / "meta" / "extra" / "uuid_map.parquet").is_file()
            ):
                uuid_map_path = dst / "meta" / "extra" / "uuid_map.parquet"
                emit_error(
                    f"--resume requested but no resume state found at {uuid_map_path}",
                    suggestion="run without --resume for the first conversion",
                    exit_code=2,
                )

            t0 = time.monotonic()
            try:
                with _silence_for_json():
                    if beta_batch:
                        convert_agibot_beta_batch(
                            src=src,
                            dst=dst,
                            max_episodes=max_episodes,
                            resume=resume,
                            workers=workers,
                        )
                    else:
                        convert_agibot_beta_to_lerobot_v3(src=src, dst=dst)
            except FileNotFoundError as exc:
                emit_error(
                    str(exc),
                    suggestion=(
                        "Beta single-episode expects proprio_stats.h5 inside the episode "
                        "dir + task_info_<task>.json at the task root; Beta batch expects "
                        "<root>/<task>/<ep_id>/proprio_stats.h5 + <root>/task_info_<task>.json"
                    ),
                    exit_code=2,
                )
            except (KeyError, ValueError) as exc:
                emit_error(
                    str(exc),
                    suggestion=(
                        "verify the Beta h5 has /state/joint/position with shape (N, 14) "
                        "and accompanying state/effector|head|waist subgroups"
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
                        "variant": "beta",
                        "episodes_written": episodes_written,
                        "frames_written": frames_written,
                        "duration_seconds": round(duration, 3),
                        "warnings": [],
                    }
                )
            if verify and not dry_run:
                _run_post_convert_verify(dst, from_format=from_format, to_format=to_format)
            return

        # variant == "digitalworld" — original sim path.
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
                    "AgiBot DigitalWorld expects parallel meta_info/<task>/<uuid>/ and "
                    "observations/<task>/<uuid>/ trees. Check that both exist under your "
                    "dataset root."
                ),
                exit_code=2,
            )
        except (KeyError, ValueError) as exc:
            emit_error(
                str(exc),
                suggestion=(
                    "verify --max-episodes is positive and that the source tree contains "
                    "discoverable episodes (proprio_states.h5 + task_info.json); "
                    "see docs/schema-agibot.md for the expected layout"
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
                    "variant": "digitalworld",
                    "episodes_written": episodes_written,
                    "frames_written": frames_written,
                    "duration_seconds": round(duration, 3),
                    "warnings": [],
                }
            )
        if verify and not dry_run:
            _run_post_convert_verify(dst, from_format=from_format, to_format=to_format)
