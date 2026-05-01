"""AgiBot Beta → LeRobot v3 forward converter (v0.2 first happy path).

Single-episode, no-video. Target user is someone with a Beta proprio HDF5
(``proprio_stats.h5``) and the sibling ``task_info_<task_id>.json`` and wants
state + action + timestamp cleanly into a LeRobot v3 directory that
``embodied-data validate`` accepts.

Schema choices documented in ``docs/v0.2-real-beta-ingest.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from embodied_data._agibot_paths import find_proprio_h5

console = Console()

JOINT_14_BETA = [
    "arm_l_j1", "arm_l_j2", "arm_l_j3", "arm_l_j4",
    "arm_l_j5", "arm_l_j6", "arm_l_j7",
    "arm_r_j1", "arm_r_j2", "arm_r_j3", "arm_r_j4",
    "arm_r_j5", "arm_r_j6", "arm_r_j7",
]  # fmt: skip

OBSERVATION_STATE_NAMES_20 = (
    JOINT_14_BETA
    + ["eff_l_width", "eff_r_width"]
    + ["head_yaw", "head_pitch"]
    + ["waist_yaw", "waist_pitch"]
)

FPS = 30
ROBOT_TYPE = "agibot-beta"
CODEBASE_VERSION = "v3.0"


def convert_agibot_beta_to_lerobot_v3(*, src: Path, dst: Path) -> None:
    """Convert one Beta episode directory to a LeRobot v3 dataset (no videos)."""
    src = Path(src)
    dst = Path(dst)

    h5_path = find_proprio_h5(src)
    if h5_path is None:
        raise FileNotFoundError(f"expected proprio_stats.h5 (or proprio_states.h5) under {src}")

    state_20, n_frames = _read_beta_state(h5_path)
    action_20 = _first_diff(state_20)
    timestamps = (np.arange(n_frames) / float(FPS)).astype(np.float32)
    frame_index = np.arange(n_frames, dtype=np.int64)
    task_name = _resolve_beta_task_name(src)

    _write_v3_dataset(
        dst=dst,
        n_frames=n_frames,
        state_20=state_20,
        action_20=action_20,
        timestamps=timestamps,
        frame_index=frame_index,
        task_name=task_name,
    )

    console.print(
        f"[green]done:[/green] {n_frames} frames → {dst} "
        f"(20-dim state + first-diff action, no videos in v0.2 first cut)"
    )


def _read_beta_state(h5_path: Path) -> tuple[np.ndarray, int]:
    """Read state subgroups → 20-dim observation: 14 joint + 2 effector + 2 head + 2 waist."""
    with h5py.File(h5_path, "r") as f:
        joint_pos = np.array(f["state/joint/position"], dtype=np.float32)
        n_frames = joint_pos.shape[0]
        if joint_pos.shape[1] != 14:
            raise ValueError(
                f"expected 14 Beta joints, got {joint_pos.shape[1]} — "
                "re-check src points at a Beta episode dir, not sim DigitalWorld"
            )
        effector_pos = _read_optional_2d(f, "state/effector/position", n_frames)
        head_pos = _read_optional_2d(f, "state/head/position", n_frames)
        waist_pos = _read_optional_2d(f, "state/waist/position", n_frames)

    state_20 = np.concatenate([joint_pos, effector_pos, head_pos, waist_pos], axis=1).astype(
        np.float32
    )
    if state_20.shape != (n_frames, 20):
        raise ValueError(f"expected (N, 20) state, got {state_20.shape}")
    return state_20, n_frames


def _read_optional_2d(f: h5py.File, path: str, n_frames: int) -> np.ndarray:
    if path in f:
        arr = np.array(f[path], dtype=np.float32)
        if arr.shape == (n_frames, 2):
            return arr
    return np.zeros((n_frames, 2), dtype=np.float32)


def _first_diff(state: np.ndarray) -> np.ndarray:
    diff = state[1:] - state[:-1]
    return np.concatenate([diff, diff[-1:]], axis=0).astype(np.float32)


def _resolve_beta_task_name(src: Path) -> str:
    """Beta task_info_<task>.json lives at the task root (parent of episode dir)."""
    for candidate_dir in (src, src.parent):
        for hit in candidate_dir.glob("task_info_*.json"):
            try:
                data = json.loads(hit.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list) and data:
                entry = data[0]
                if isinstance(entry, dict):
                    name = entry.get("task_name") or entry.get("task")
                    if name:
                        return str(name)
            elif isinstance(data, dict):
                name = data.get("task_name")
                if name:
                    return str(name)
    sim_path = src / "task_info.json"
    if sim_path.is_file():
        try:
            data = json.loads(sim_path.read_text())
            if isinstance(data, dict):
                return str(data.get("task_name") or "unknown")
        except (OSError, json.JSONDecodeError):
            pass
    return "unknown"


def _write_v3_dataset(
    *,
    dst: Path,
    n_frames: int,
    state_20: np.ndarray,
    action_20: np.ndarray,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
    task_name: str,
) -> None:
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    _write_info_json(dst, n_frames=n_frames)
    _write_tasks_parquet(dst, task_name=task_name)
    _write_data_parquet(
        dst,
        n_frames=n_frames,
        state_20=state_20,
        action_20=action_20,
        timestamps=timestamps,
        frame_index=frame_index,
    )
    _write_episodes_parquet(
        dst,
        n_frames=n_frames,
        task_name=task_name,
        state_20=state_20,
        action_20=action_20,
    )
    _write_stats_json(dst, state_20=state_20, action_20=action_20)


def _stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def _fixed_size_list(arr: np.ndarray, dim: int) -> pa.Array:
    flat = pa.array(arr.reshape(-1).tolist(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _write_info_json(dst: Path, *, n_frames: int) -> None:
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "total_episodes": 1,
        "total_frames": int(n_frames),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{n_frames}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [20],
                "names": list(OBSERVATION_STATE_NAMES_20),
            },
            "action": {
                "dtype": "float32",
                "shape": [20],
                "names": list(OBSERVATION_STATE_NAMES_20),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _write_tasks_parquet(dst: Path, *, task_name: str) -> None:
    table = pa.table(
        {
            "task_index": pa.array([0], type=pa.int64()),
            "task": pa.array([task_name], type=pa.string()),
        }
    )
    pq.write_table(table, dst / "meta" / "tasks.parquet")


def _write_data_parquet(
    dst: Path,
    *,
    n_frames: int,
    state_20: np.ndarray,
    action_20: np.ndarray,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
) -> None:
    zeros = np.zeros(n_frames, dtype=np.int64)
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(zeros, type=pa.int64()),
            "index": pa.array(frame_index, type=pa.int64()),
            "task_index": pa.array(zeros, type=pa.int64()),
            "observation.state": _fixed_size_list(state_20, 20),
            "action": _fixed_size_list(action_20, 20),
        }
    )
    pq.write_table(table, dst / "data" / "chunk-000" / "file-000.parquet")


def _write_episodes_parquet(
    dst: Path,
    *,
    n_frames: int,
    task_name: str,
    state_20: np.ndarray,
    action_20: np.ndarray,
) -> None:
    s = _stats(state_20)
    a = _stats(action_20)
    row = {
        "episode_index": [0],
        "tasks": [[task_name]],
        "length": [int(n_frames)],
        "meta/episodes/chunk_index": [0],
        "meta/episodes/file_index": [0],
        "data/chunk_index": [0],
        "data/file_index": [0],
        "dataset_from_index": [0],
        "dataset_to_index": [int(n_frames)],
        "stats/observation.state/min": [s["min"]],
        "stats/observation.state/max": [s["max"]],
        "stats/observation.state/mean": [s["mean"]],
        "stats/observation.state/std": [s["std"]],
        "stats/observation.state/count": [s["count"]],
        "stats/action/min": [a["min"]],
        "stats/action/max": [a["max"]],
        "stats/action/mean": [a["mean"]],
        "stats/action/std": [a["std"]],
        "stats/action/count": [a["count"]],
    }
    table = pa.Table.from_pydict(row)
    pq.write_table(table, dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def _write_stats_json(dst: Path, *, state_20: np.ndarray, action_20: np.ndarray) -> None:
    payload = {
        "observation.state": _stats(state_20),
        "action": _stats(action_20),
    }
    (dst / "meta" / "stats.json").write_text(json.dumps(payload, indent=2))
