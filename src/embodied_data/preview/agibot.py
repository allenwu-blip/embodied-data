from __future__ import annotations

import json
from pathlib import Path

import av
import h5py

from embodied_data._agibot_paths import find_proprio_h5

Stat = tuple[str, str]

# Per docs/schema-agibot.md §3 the official converter keeps 22 of 34 raw joints.
AGIBOT_KEPT_JOINTS = 22


def _find_video_dir(episode_dir: Path) -> Path | None:
    """Locate observations/<task>/<uuid>/video/ given a meta_info episode dir.

    Layout: <root>/meta_info/<task>/<uuid>/  ↔  <root>/observations/<task>/<uuid>/video/.
    Falls back to None if the sibling tree is absent.
    """
    parts = episode_dir.parts
    if "meta_info" not in parts:
        return None
    idx = parts.index("meta_info")
    root = Path(*parts[:idx])
    tail = parts[idx + 1 :]
    candidate = root / "observations" / Path(*tail) / "video"
    return candidate if candidate.is_dir() else None


def _read_first_mp4_fps(video_dir: Path) -> tuple[int | None, int | None]:
    """Return (fps, frame_count) from the first readable mp4, else (None, None)."""
    for mp4 in sorted(video_dir.glob("*.mp4")):
        try:
            with av.open(str(mp4)) as container:
                stream = container.streams.video[0]
                rate = stream.average_rate
                fps = int(round(float(rate))) if rate else None
                frames = int(stream.frames) if stream.frames else None
                return fps, frames
        except (av.FFmpegError, OSError):
            continue
    return None, None


def _read_task_name(episode_dir: Path) -> str | None:
    task_path = episode_dir / "task_info.json"
    if not task_path.is_file():
        return None
    try:
        obj = json.loads(task_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    name = obj.get("task_name")
    return str(name) if name else None


def _read_h5_dims(h5_path: Path) -> tuple[int, int, int, str]:
    """Return (frames, raw_state_dim, action_dim, robot_type) from proprio_states.h5."""
    with h5py.File(h5_path, "r") as f:
        if "state/joint/position" not in f:
            raise ValueError("h5 missing 'state/joint/position'")
        state_pos = f["state/joint/position"]
        frames = int(state_pos.shape[0])
        raw_state_dim = int(state_pos.shape[1]) if state_pos.ndim == 2 else 1

        action_dim = raw_state_dim
        if "action/joint/position" in f:
            action_pos = f["action/joint/position"]
            if action_pos.ndim == 2:
                action_dim = int(action_pos.shape[1])

        robot_type = "a2d"
        robot_attrs = f.get("state/robot")
        if robot_attrs is not None and "name" in robot_attrs.attrs:
            names = robot_attrs.attrs["name"]
            if hasattr(names, "__len__") and len(names) > 0:
                robot_type = str(names[0]).lower()
    return frames, raw_state_dim, action_dim, robot_type


def collect_agibot_stats(path: Path, n: int) -> tuple[list[Stat], str]:
    """Read a single AgiBot episode dir. n is accepted for signature parity; AgiBot
    meta_info layout is one episode per dir, so we never truncate."""
    del n
    h5_path = find_proprio_h5(path)
    video_dir = _find_video_dir(path)

    h5_present = h5_path is not None
    if not h5_present and video_dir is None:
        raise ValueError(f"agibot episode dir missing both proprio_state[s]*.h5 and videos: {path}")

    frames = raw_state_dim = action_dim = 0
    robot_type = "a2d"
    if h5_present:
        frames, raw_state_dim, action_dim, robot_type = _read_h5_dims(h5_path)

    fps: int | None = None
    video_frames: int | None = None
    cameras: list[str] = []
    if video_dir is not None:
        fps, video_frames = _read_first_mp4_fps(video_dir)
        cameras = sorted(p.stem for p in video_dir.glob("*.mp4"))

    if not frames and video_frames:
        frames = video_frames
    if fps is None or fps <= 0:
        # AgiBot mp4s are documented as 30 fps; fall back rather than crash on
        # a partial dir without videos.
        fps = 30
    duration_s = frames / fps if fps else 0.0

    task_name = _read_task_name(path) or "(unknown)"

    state_dim_text = (
        f"{AGIBOT_KEPT_JOINTS} (raw {raw_state_dim})" if raw_state_dim else str(AGIBOT_KEPT_JOINTS)
    )

    # AgiBot meta_info is one episode per directory. We never truncate.
    header = ""

    stats: list[Stat] = [
        ("Format", "agibot"),
        ("Episodes", "1 sampled / 1 total"),
        ("Total frames", str(frames)),
        ("Total duration", f"{duration_s:.1f}s"),
        ("fps", str(fps)),
        ("robot_type", robot_type),
        ("State dim", state_dim_text),
        ("Action dim", str(action_dim) if action_dim else "?"),
        ("Cameras", ", ".join(cameras) if cameras else "(none)"),
        ("Tasks", f'1: "{task_name}"'),
    ]
    return stats, header
