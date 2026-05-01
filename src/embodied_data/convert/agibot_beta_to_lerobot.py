"""AgiBot Beta → LeRobot v3 forward converter (v0.2 M1 single + M3 batch).

Beta layout differs from sim DigitalWorld in five places (see
``docs/v0.1.1-findings.md``): filename is ``proprio_stats.h5`` (typo that
shipped); ``state/joint/position`` is ``(N, 14)`` not ``(N, 34)`` and lacks
the ``name`` attr; extra ``state/{head,waist}`` subgroups; ``/timestamp`` is
``int64`` ns Unix-epoch (we discard and recompute); ``task_info_<task>.json``
sits at the task-dataset root holding a list of per-episode dicts.

Single-episode (M1): ``convert_agibot_beta_to_lerobot_v3`` writes a complete
v3 dataset into ``dst`` from one episode dir. No videos are emitted (the
upstream Beta sample we hold ships proprio + metadata only).

Multi-episode (M3): ``convert_agibot_beta_batch`` discovers every
``<task>/<ep_id>/proprio_stats.h5`` under ``src``, supports ``--max-episodes``
/ ``--resume`` / ``--workers``, and writes a single multi-episode v3 dataset.
Idempotent on rerun via ``meta/extra/uuid_map.parquet``. Failed episodes are
logged to ``<dst>/.beta_batch_errors.jsonl`` and the run continues.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

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


# ---------------------------------------------------------------------------
# Multi-episode batch (M3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BetaEpisodeSource:
    """One Beta episode discovered under a task-dataset root."""

    task: str  # e.g. '675' (task id, also the dir name in <root>/<task>/<ep>/)
    episode_id: str  # e.g. '936938'
    h5_path: Path
    task_info_path: Path  # canonical: <root>/task_info_<task>.json

    @property
    def key(self) -> tuple[str, str]:
        return (self.task, self.episode_id)


@dataclass
class _BetaPerEpisodePayload:
    """Worker output. Small (state arrays + paths only) — safe to pickle."""

    state_20: np.ndarray
    action_20: np.ndarray
    task_name: str
    n_frames: int


def is_beta_batch_src(src: Path) -> bool:
    """Heuristic: src looks like a Beta task-dataset root (has task_info_*.json
    sibling + at least one ``<task>/<ep>/proprio_stats.h5`` in subtree)."""
    src = Path(src)
    if not src.is_dir():
        return False
    if find_proprio_h5(src) is not None:
        return False  # this IS a single-episode dir, not a batch root
    if not any(src.glob("task_info_*.json")):
        return False
    return any(src.rglob("proprio_stats.h5"))


def _discover_beta_episodes(src: Path) -> list[_BetaEpisodeSource]:
    """Walk ``<src>/<task>/<ep_id>/proprio_stats.h5`` and resolve each episode's
    task_info file. Skips paths that don't match the canonical Beta layout."""
    out: list[_BetaEpisodeSource] = []
    for h5 in src.rglob("proprio_stats.h5"):
        if not h5.is_file():
            continue
        ep_dir = h5.parent
        task_dir = ep_dir.parent
        if task_dir == src or not task_dir.is_relative_to(src):
            continue
        task = task_dir.name
        episode_id = ep_dir.name
        # Canonical: task_info at <root>/task_info_<task>.json.
        # Fallbacks: <task_dir>/task_info_<task>.json, <ep_dir>/task_info.json.
        candidates = [
            src / f"task_info_{task}.json",
            task_dir / f"task_info_{task}.json",
            ep_dir / "task_info.json",
        ]
        task_info_path = next((c for c in candidates if c.is_file()), None)
        if task_info_path is None:
            continue
        out.append(
            _BetaEpisodeSource(
                task=task,
                episode_id=episode_id,
                h5_path=h5,
                task_info_path=task_info_path,
            )
        )
    return out


def _resolve_task_name_from_file(task_info_path: Path) -> str:
    """Extract task name from a Beta-layout task_info JSON. Tolerates the
    list-of-episodes format (uses [0]['task_name']) and the sim single-dict
    format. Falls back to 'unknown' on any read/parse error."""
    if not task_info_path.is_file():
        return "unknown"
    try:
        data = json.loads(task_info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("task_name") or data[0].get("task") or "unknown")
    if isinstance(data, dict):
        return str(data.get("task_name") or "unknown")
    return "unknown"


def _load_beta_episode(src_ep: _BetaEpisodeSource) -> _BetaPerEpisodePayload:
    """Worker: read one Beta episode's h5 + task_info into a payload."""
    state_20, n_frames = _read_beta_state(src_ep.h5_path)
    action_20 = _first_diff(state_20)
    task_name = _resolve_task_name_from_file(src_ep.task_info_path)
    return _BetaPerEpisodePayload(
        state_20=state_20,
        action_20=action_20,
        task_name=task_name,
        n_frames=n_frames,
    )


def _commit_beta_episode(
    *,
    dst: Path,
    payload: _BetaPerEpisodePayload,
    src_ep: _BetaEpisodeSource,
    episode_index: int,
    task_to_index: dict[str, int],
    uuid_map_rows: list[dict],
    state_chunks: list[np.ndarray],
    action_chunks: list[np.ndarray],
) -> int:
    """Write one episode's data + episode-meta parquet rows. Returns next ep idx.

    Beta has no videos and ships small parquets, so we use one-episode-per-file
    naming inside chunk-000 (file-NNN.parquet) and never roll the chunk dir.
    """
    if payload.task_name not in task_to_index:
        task_to_index[payload.task_name] = len(task_to_index)
    task_index = task_to_index[payload.task_name]

    dataset_from_index = sum(int(arr.shape[0]) for arr in state_chunks)
    file_index = episode_index  # one episode per file inside chunk-000

    timestamps = (np.arange(payload.n_frames) / float(FPS)).astype(np.float32)
    frame_index = np.arange(payload.n_frames, dtype=np.int64)
    episode_index_arr = np.full(payload.n_frames, episode_index, dtype=np.int64)
    task_index_arr = np.full(payload.n_frames, task_index, dtype=np.int64)
    global_index_arr = np.arange(
        dataset_from_index, dataset_from_index + payload.n_frames, dtype=np.int64
    )

    data_dir = dst / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"file-{file_index:03d}.parquet"
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index_arr, type=pa.int64()),
            "index": pa.array(global_index_arr, type=pa.int64()),
            "task_index": pa.array(task_index_arr, type=pa.int64()),
            "observation.state": _fixed_size_list(payload.state_20, 20),
            "action": _fixed_size_list(payload.action_20, 20),
        }
    )
    pq.write_table(table, data_path)

    ep_meta_dir = dst / "meta" / "episodes" / "chunk-000"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_path = ep_meta_dir / f"file-{file_index:03d}.parquet"
    s_state = _stats(payload.state_20)
    s_action = _stats(payload.action_20)
    row = {
        "episode_index": [episode_index],
        "tasks": [[payload.task_name]],
        "length": [int(payload.n_frames)],
        "meta/episodes/chunk_index": [0],
        "meta/episodes/file_index": [file_index],
        "data/chunk_index": [0],
        "data/file_index": [file_index],
        "dataset_from_index": [dataset_from_index],
        "dataset_to_index": [dataset_from_index + int(payload.n_frames)],
        "stats/observation.state/min": [s_state["min"]],
        "stats/observation.state/max": [s_state["max"]],
        "stats/observation.state/mean": [s_state["mean"]],
        "stats/observation.state/std": [s_state["std"]],
        "stats/observation.state/count": [s_state["count"]],
        "stats/action/min": [s_action["min"]],
        "stats/action/max": [s_action["max"]],
        "stats/action/mean": [s_action["mean"]],
        "stats/action/std": [s_action["std"]],
        "stats/action/count": [s_action["count"]],
    }
    pq.write_table(pa.Table.from_pydict(row), ep_meta_path)

    state_chunks.append(payload.state_20)
    action_chunks.append(payload.action_20)
    uuid_map_rows.append(
        {
            "episode_index": episode_index,
            "agibot_task": src_ep.task,
            "agibot_episode_id": src_ep.episode_id,
        }
    )
    return episode_index + 1


# ---------------------------------------------------------------------------
# Resume / aggregate helpers.
# ---------------------------------------------------------------------------


def _load_beta_resume_state(dst: Path) -> tuple[set[tuple[str, str]], int]:
    """Read ``meta/extra/uuid_map.parquet`` → (already-done keys, next ep idx)."""
    uuid_map = dst / "meta" / "extra" / "uuid_map.parquet"
    if not uuid_map.is_file():
        return set(), 0
    table = pq.read_table(uuid_map)
    done: set[tuple[str, str]] = set()
    max_ep = -1
    for task, ep_id, ep in zip(
        table["agibot_task"].to_pylist(),
        table["agibot_episode_id"].to_pylist(),
        table["episode_index"].to_pylist(),
        strict=True,
    ):
        done.add((str(task), str(ep_id)))
        if ep > max_ep:
            max_ep = ep
    return done, max_ep + 1


def _load_existing_beta_arrays(dst: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Re-load on-disk per-episode state/action arrays so resume produces
    byte-identical info.json/stats.json as a clean run."""
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    data_dir = dst / "data" / "chunk-000"
    if not data_dir.is_dir():
        return state_chunks, action_chunks
    for parquet in sorted(data_dir.glob("file-*.parquet")):
        table = pq.read_table(parquet, columns=["observation.state", "action"])
        state_chunks.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        action_chunks.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
    return state_chunks, action_chunks


def _load_beta_task_index(dst: Path) -> dict[str, int]:
    tasks_path = dst / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}
    table = pq.read_table(tasks_path)
    return {
        str(task): int(idx)
        for task, idx in zip(
            table["task"].to_pylist(),
            table["task_index"].to_pylist(),
            strict=True,
        )
    }


def _load_beta_uuid_map(dst: Path) -> list[dict]:
    uuid_map = dst / "meta" / "extra" / "uuid_map.parquet"
    if not uuid_map.is_file():
        return []
    table = pq.read_table(uuid_map)
    return table.to_pylist()


def _write_beta_tasks_parquet_multi(dst: Path, task_to_index: dict[str, int]) -> None:
    if not task_to_index:
        return
    tasks = [t for t, _ in sorted(task_to_index.items(), key=lambda kv: kv[1])]
    indices = list(range(len(tasks)))
    table = pa.table(
        {
            "task_index": pa.array(indices, type=pa.int64()),
            "task": pa.array(tasks, type=pa.string()),
        }
    )
    pq.write_table(table, dst / "meta" / "tasks.parquet")


def _write_beta_uuid_map(dst: Path, rows: list[dict]) -> None:
    if not rows:
        return
    table = pa.table(
        {
            "episode_index": pa.array([int(r["episode_index"]) for r in rows], type=pa.int64()),
            "agibot_task": pa.array([str(r["agibot_task"]) for r in rows], type=pa.string()),
            "agibot_episode_id": pa.array(
                [str(r["agibot_episode_id"]) for r in rows], type=pa.string()
            ),
        }
    )
    pq.write_table(table, dst / "meta" / "extra" / "uuid_map.parquet")


def _write_beta_stats_json_multi(
    dst: Path,
    *,
    state_chunks: list[np.ndarray],
    action_chunks: list[np.ndarray],
) -> None:
    if not state_chunks:
        return
    state_all = np.concatenate(state_chunks, axis=0)
    action_all = np.concatenate(action_chunks, axis=0)
    payload = {
        "observation.state": _stats(state_all),
        "action": _stats(action_all),
    }
    (dst / "meta" / "stats.json").write_text(json.dumps(payload, indent=2))


def _write_beta_info_json_multi(
    dst: Path,
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{total_frames}"},
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


def _finalize_beta_aggregates(dst: Path) -> None:
    """Re-emit info.json / stats.json from on-disk truth (used on no-op resume)."""
    state_chunks, action_chunks = _load_existing_beta_arrays(dst)
    if not state_chunks:
        return
    total_frames = sum(arr.shape[0] for arr in state_chunks)
    task_to_index = _load_beta_task_index(dst)
    _write_beta_stats_json_multi(dst, state_chunks=state_chunks, action_chunks=action_chunks)
    _write_beta_info_json_multi(
        dst,
        total_episodes=len(state_chunks),
        total_frames=total_frames,
        total_tasks=len(task_to_index),
    )


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object as a line to ``path`` (creates parent if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj))
        f.write("\n")


def convert_agibot_beta_batch(
    *,
    src: Path,
    dst: Path,
    max_episodes: int | None = None,
    resume: bool = False,
    workers: int = 1,
) -> None:
    """Convert many Beta episodes under ``src`` into one LeRobot v3 dataset.

    ``src`` must be a Beta task-dataset root whose subtree contains
    ``<task>/<ep_id>/proprio_stats.h5`` files plus ``task_info_<task>.json``
    at the root level.
    """
    src = Path(src)
    dst = Path(dst)

    sources = _discover_beta_episodes(src)
    if not sources:
        raise FileNotFoundError(
            f"no Beta episodes found under {src} "
            "(expected <task>/<ep_id>/proprio_stats.h5 + task_info_<task>.json at root)"
        )
    sources.sort(key=lambda s: s.key)
    if max_episodes is not None:
        sources = sources[: max(0, max_episodes)]
    if not sources:
        raise ValueError("max_episodes truncated source list to zero")

    done_keys: set[tuple[str, str]] = set()
    next_episode_index = 0
    if resume:
        done_keys, next_episode_index = _load_beta_resume_state(dst)
    elif dst.exists():
        shutil.rmtree(dst)

    pending = [s for s in sources if s.key not in done_keys]
    if not pending:
        console.print(
            f"[green]nothing to do:[/green] all {len(sources)} Beta episodes already converted"
        )
        _finalize_beta_aggregates(dst)
        return

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "extra").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    if resume:
        state_chunks, action_chunks = _load_existing_beta_arrays(dst)
    task_to_index: dict[str, int] = _load_beta_task_index(dst) if resume else {}
    uuid_map_rows: list[dict] = _load_beta_uuid_map(dst) if resume else []

    error_log_path = dst / ".beta_batch_errors.jsonl"
    if not resume and error_log_path.exists():
        error_log_path.unlink()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• {task.fields[stage]}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task(
        f"converting {len(pending)} Beta episodes",
        total=len(pending),
        stage="starting",
    )
    succeeded = 0
    failed = 0

    def commit_one(src_ep: _BetaEpisodeSource, payload: _BetaPerEpisodePayload) -> None:
        nonlocal next_episode_index, succeeded
        next_episode_index = _commit_beta_episode(
            dst=dst,
            payload=payload,
            src_ep=src_ep,
            episode_index=next_episode_index,
            task_to_index=task_to_index,
            uuid_map_rows=uuid_map_rows,
            state_chunks=state_chunks,
            action_chunks=action_chunks,
        )
        succeeded += 1

    with progress:
        if workers <= 1:
            for src_ep in pending:
                progress.update(
                    task_id, stage=f"ep {next_episode_index}: {src_ep.task}/{src_ep.episode_id}"
                )
                try:
                    payload = _load_beta_episode(src_ep)
                except (OSError, KeyError, ValueError) as exc:
                    failed += 1
                    _append_jsonl(
                        error_log_path,
                        {
                            "task": src_ep.task,
                            "episode_id": src_ep.episode_id,
                            "h5_path": str(src_ep.h5_path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    progress.update(task_id, advance=1)
                    continue
                commit_one(src_ep, payload)
                progress.update(task_id, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_load_beta_episode, s) for s in pending]
                for src_ep, fut in zip(pending, futures, strict=True):
                    progress.update(
                        task_id,
                        stage=f"ep {next_episode_index}: {src_ep.task}/{src_ep.episode_id}",
                    )
                    try:
                        payload = fut.result()
                    except (OSError, KeyError, ValueError) as exc:
                        failed += 1
                        _append_jsonl(
                            error_log_path,
                            {
                                "task": src_ep.task,
                                "episode_id": src_ep.episode_id,
                                "h5_path": str(src_ep.h5_path),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        progress.update(task_id, advance=1)
                        continue
                    commit_one(src_ep, payload)
                    progress.update(task_id, advance=1)

    _write_beta_tasks_parquet_multi(dst, task_to_index)
    _write_beta_uuid_map(dst, uuid_map_rows)
    disk_state, disk_action = _load_existing_beta_arrays(dst)
    _write_beta_stats_json_multi(dst, state_chunks=disk_state, action_chunks=disk_action)
    total_frames_written = sum(arr.shape[0] for arr in disk_state)
    _write_beta_info_json_multi(
        dst,
        total_episodes=next_episode_index,
        total_frames=total_frames_written,
        total_tasks=len(task_to_index),
    )

    summary = (
        f"[green]done:[/green] {succeeded} succeeded, "
        f"{failed} failed → {dst} ({total_frames_written} frames, "
        f"{len(task_to_index)} task(s))"
    )
    if failed:
        summary += f" — see {error_log_path.name}"
    console.print(summary)
