from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
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

from embodied_data._agibot_paths import PROPRIO_GLOB, find_proprio_h5

console = Console()

# 22-joint subselection order from upstream script:272-308.
# Effector sub-order is right-then-left to match official output.
HEAD_JOINTS = ["joint_head_yaw", "joint_head_pitch"]
BODY_JOINTS = ["joint_lift_body", "joint_body_pitch"]
ARM_JOINTS = [
    "Joint1_l", "Joint1_r", "Joint2_l", "Joint2_r",
    "Joint3_l", "Joint3_r", "Joint4_l", "Joint4_r",
    "Joint5_l", "Joint5_r", "Joint6_l", "Joint6_r",
    "Joint7_l", "Joint7_r",
]  # fmt: skip
EFFECTOR_JOINTS = [
    "right_Left_1_Joint", "right_Right_1_Joint",
    "left_Left_1_Joint", "left_Right_1_Joint",
]  # fmt: skip
JOINT_22 = HEAD_JOINTS + BODY_JOINTS + ARM_JOINTS + EFFECTOR_JOINTS

FPS = 30
ROBOT_TYPE = "a2d"
HEAD_CAMERA_KEY = "observation.images.top_head"

# Sprint 2 chunk-rollover policy: one episode per parquet/mp4 file. Multiple
# files share a chunk directory until cumulative bytes exceed the v3 limits
# (100MB parquet, 200MB video). Episodes are never split across files.
CHUNKS_SIZE = 1000
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 200


# ---------------------------------------------------------------------------
# Single-episode public API (preserved verbatim for backwards compat).
# ---------------------------------------------------------------------------


def convert_agibot_to_lerobot_v3(*, src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)

    h5_path, task_info_path, head_video = _resolve_inputs(src)
    _assert_digitalworld_sim(h5_path)

    state_22, n_frames, _ = _read_state(h5_path)
    action_22 = _first_diff_action(state_22)
    task_name = _read_task_name(task_info_path)

    timestamps = (np.arange(n_frames) / float(FPS)).astype(np.float32)
    frame_index = np.arange(n_frames, dtype=np.int64)

    _write_dataset_single(
        dst=dst,
        n_frames=n_frames,
        state_22=state_22,
        action_22=action_22,
        timestamps=timestamps,
        frame_index=frame_index,
        task_name=task_name,
        head_video=head_video,
    )

    console.print(
        f"[green]done:[/green] {n_frames} frames, 1 task, 1 camera ({HEAD_CAMERA_KEY}) → {dst}"
    )


# ---------------------------------------------------------------------------
# Batch (multi-episode) public API.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EpisodeSource:
    """One AgiBot episode discovered by recursive glob."""

    task: str  # e.g. 'digitaltwin_3'
    uuid: str  # e.g. '000aa0b4-...'
    h5_path: Path
    task_info_path: Path
    head_video: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.task, self.uuid)


@dataclass
class _PerEpisodePayload:
    """Worker output. Small (state arrays + paths only) — safe to pickle."""

    state_22: np.ndarray
    action_22: np.ndarray
    task_name: str
    n_frames: int
    head_video: Path  # source mp4; main process re-encodes to dst


def convert_agibot_batch(
    *,
    src: Path,
    dst: Path,
    max_episodes: int | None = None,
    resume: bool = False,
    workers: int = 1,
) -> None:
    """Convert many AgiBot episodes under ``src`` into one LeRobot v3 dataset.

    ``src`` must be a directory whose subtree contains
    ``<task>/<uuid>/proprio_states.h5`` files (e.g. ``meta_info/``).
    """
    src = Path(src)
    dst = Path(dst)

    sources = _discover_episodes(src)
    if not sources:
        raise FileNotFoundError(
            f"no AgiBot episodes found under {src} (expected **/<task>/<uuid>/proprio_state[s]*.h5)"
        )
    sources.sort(key=lambda s: s.key)
    if max_episodes is not None:
        sources = sources[: max(0, max_episodes)]
    if not sources:
        raise ValueError("max_episodes truncated source list to zero")

    # Resume bookkeeping: read the uuid map (if any) to find UUIDs already done.
    done_uuids: set[tuple[str, str]] = set()
    next_episode_index = 0
    next_chunk_index = 0
    next_file_index = 0
    cum_data_bytes = 0
    cum_video_bytes = 0
    if resume:
        (
            done_uuids,
            next_episode_index,
            next_chunk_index,
            next_file_index,
            cum_data_bytes,
            cum_video_bytes,
        ) = _load_resume_state(dst)
    else:
        # Clean slate. Only nuke if the directory exists; new dirs are fine.
        if dst.exists():
            shutil.rmtree(dst)

    # Filter out already-done episodes.
    pending = [s for s in sources if s.key not in done_uuids]
    if not pending:
        console.print(
            f"[green]nothing to do:[/green] all {len(sources)} episodes already converted"
        )
        # Still re-emit info.json/stats.json to reflect the current state on disk.
        _finalize_aggregates(dst)
        return

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "extra").mkdir(parents=True, exist_ok=True)

    # Aggregate stats accumulators. Re-load from existing on-disk episodes
    # parquets if resuming so that final stats.json is identical.
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    if resume:
        state_chunks, action_chunks = _load_existing_arrays(dst)

    task_to_index: dict[str, int] = _load_task_index(dst) if resume else {}
    uuid_map_rows: list[dict] = _load_uuid_map(dst) if resume else []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• {task.fields[stage]}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task(
        f"converting {len(pending)} episodes",
        total=len(pending),
        stage="starting",
    )
    total_frames_written = sum(arr.shape[0] for arr in state_chunks)

    with progress:
        if workers <= 1:
            iterable = (_load_episode(src_ep) for src_ep in pending)
            for src_ep, payload in zip(pending, iterable, strict=True):
                stage = f"ep {next_episode_index}: {src_ep.uuid[:8]}"
                stage += f" ({total_frames_written} frames)"
                progress.update(task_id, stage=stage)
                (
                    next_episode_index,
                    next_chunk_index,
                    next_file_index,
                    cum_data_bytes,
                    cum_video_bytes,
                ) = _commit_episode(
                    dst=dst,
                    payload=payload,
                    src_ep=src_ep,
                    episode_index=next_episode_index,
                    chunk_index=next_chunk_index,
                    file_index=next_file_index,
                    cum_data_bytes=cum_data_bytes,
                    cum_video_bytes=cum_video_bytes,
                    task_to_index=task_to_index,
                    uuid_map_rows=uuid_map_rows,
                    state_chunks=state_chunks,
                    action_chunks=action_chunks,
                )
                total_frames_written += payload.n_frames
                progress.update(task_id, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # Submit all up front but commit in deterministic source order.
                futures = [pool.submit(_load_episode, s) for s in pending]
                for src_ep, fut in zip(pending, futures, strict=True):
                    payload = fut.result()
                    stage = f"ep {next_episode_index}: {src_ep.uuid[:8]}"
                    stage += f" ({total_frames_written} frames)"
                    progress.update(task_id, stage=stage)
                    (
                        next_episode_index,
                        next_chunk_index,
                        next_file_index,
                        cum_data_bytes,
                        cum_video_bytes,
                    ) = _commit_episode(
                        dst=dst,
                        payload=payload,
                        src_ep=src_ep,
                        episode_index=next_episode_index,
                        chunk_index=next_chunk_index,
                        file_index=next_file_index,
                        cum_data_bytes=cum_data_bytes,
                        cum_video_bytes=cum_video_bytes,
                        task_to_index=task_to_index,
                        uuid_map_rows=uuid_map_rows,
                        state_chunks=state_chunks,
                        action_chunks=action_chunks,
                    )
                    total_frames_written += payload.n_frames
                    progress.update(task_id, advance=1)

    # Finalize aggregate artifacts. Stats are recomputed from on-disk parquets
    # (not the in-memory accumulators) so the resume path produces identical
    # bytes — both paths see the same float32→float64→float32 round-trip.
    _write_tasks_parquet_multi(dst, task_to_index)
    _write_uuid_map(dst, uuid_map_rows)
    disk_state, disk_action = _load_existing_arrays(dst)
    _write_stats_json_multi(dst, state_chunks=disk_state, action_chunks=disk_action)
    _write_info_json_multi(
        dst,
        total_episodes=next_episode_index,
        total_frames=total_frames_written,
        total_tasks=len(task_to_index),
    )

    console.print(
        f"[green]done:[/green] {next_episode_index} episodes, "
        f"{total_frames_written} frames, {len(task_to_index)} task(s) → {dst}"
    )


# ---------------------------------------------------------------------------
# Discovery / dispatch.
# ---------------------------------------------------------------------------


def is_batch_src(src: Path) -> bool:
    """Heuristic: src is batch if its own dir lacks any proprio_state[s]*.h5."""
    return find_proprio_h5(Path(src)) is None


def _discover_episodes(src: Path) -> list[_EpisodeSource]:
    out: list[_EpisodeSource] = []
    skipped_beta_like = 0  # h5 found but layout looks like real Beta, not sim
    for h5 in src.rglob(PROPRIO_GLOB):
        if not h5.is_file():
            continue
        ep_dir = h5.parent
        ep_dir_str = str(ep_dir)
        task_info = ep_dir / "task_info.json"
        # Beta sample uses task_info_<task_id>.json one level up + no /meta_info/ prefix.
        # Detect both Beta layout markers so we can raise a clear refusal below.
        looks_beta = (
            "/meta_info/" not in ep_dir_str
            or not task_info.is_file()
            and any(ep_dir.parent.glob("task_info_*.json"))
        )
        if not task_info.is_file():
            if looks_beta:
                skipped_beta_like += 1
            continue
        # Recover (task, uuid). Layout is .../<task>/<uuid>/proprio_state[s]*.h5.
        try:
            task = ep_dir.parent.name
            uuid = ep_dir.name
        except IndexError:
            continue
        # Sibling video path lives under .../observations/<task>/<uuid>/video/head.mp4.
        if "/meta_info/" not in ep_dir_str:
            skipped_beta_like += 1
            continue
        obs_dir = Path(ep_dir_str.replace("/meta_info/", "/observations/", 1))
        head_video = obs_dir / "video" / "head.mp4"
        if not head_video.is_file():
            continue
        out.append(
            _EpisodeSource(
                task=task,
                uuid=uuid,
                h5_path=h5,
                task_info_path=task_info,
                head_video=head_video,
            )
        )
    if not out and skipped_beta_like:
        raise ValueError(
            f"detected {skipped_beta_like} real-robot AgiBot capture(s) under {src} "
            "(Beta-style layout: no /meta_info/ prefix and/or task_info_<task>.json "
            "instead of task_info.json). embodied-data v0.1 supports DigitalWorld sim "
            "only; Beta/Alpha forward conversion is on the v0.2 roadmap."
        )
    return out


# ---------------------------------------------------------------------------
# Worker-side: per-episode load (h5 only; video is re-encoded in main proc).
# ---------------------------------------------------------------------------


def _load_episode(src_ep: _EpisodeSource) -> _PerEpisodePayload:
    _assert_digitalworld_sim(src_ep.h5_path)
    state_22, n_frames, _ = _read_state(src_ep.h5_path)
    action_22 = _first_diff_action(state_22)
    task_name = _read_task_name(src_ep.task_info_path)
    return _PerEpisodePayload(
        state_22=state_22,
        action_22=action_22,
        task_name=task_name,
        n_frames=n_frames,
        head_video=src_ep.head_video,
    )


# ---------------------------------------------------------------------------
# Per-episode commit (always main-process; serializes parquet writes).
# ---------------------------------------------------------------------------


def _commit_episode(
    *,
    dst: Path,
    payload: _PerEpisodePayload,
    src_ep: _EpisodeSource,
    episode_index: int,
    chunk_index: int,
    file_index: int,
    cum_data_bytes: int,
    cum_video_bytes: int,
    task_to_index: dict[str, int],
    uuid_map_rows: list[dict],
    state_chunks: list[np.ndarray],
    action_chunks: list[np.ndarray],
) -> tuple[int, int, int, int, int]:
    """Write one episode to disk; return updated indices and cumulative bytes."""
    # Reserve task index.
    if payload.task_name not in task_to_index:
        task_to_index[payload.task_name] = len(task_to_index)
    task_index = task_to_index[payload.task_name]

    # Compute global frame offset before this episode.
    dataset_from_index = sum(int(arr.shape[0]) for arr in state_chunks)

    # --- estimate next-file size; rotate if needed ---
    # 32 bytes per row is approximate (state+action 22f32 = 176B + ~80B scalars).
    est_data_bytes = payload.n_frames * 256
    est_video_bytes = payload.head_video.stat().st_size  # close enough for h264 re-encode
    if cum_data_bytes + est_data_bytes > DATA_FILE_SIZE_MB * 1024 * 1024 and file_index > 0:
        file_index += 1
        cum_data_bytes = 0
        cum_video_bytes = 0
    if cum_video_bytes + est_video_bytes > VIDEO_FILE_SIZE_MB * 1024 * 1024 and file_index > 0:
        file_index += 1
        cum_data_bytes = 0
        cum_video_bytes = 0
    if file_index >= CHUNKS_SIZE:
        chunk_index += 1
        file_index = 0
        cum_data_bytes = 0
        cum_video_bytes = 0

    timestamps = (np.arange(payload.n_frames) / float(FPS)).astype(np.float32)
    frame_index = np.arange(payload.n_frames, dtype=np.int64)
    episode_index_arr = np.full(payload.n_frames, episode_index, dtype=np.int64)
    task_index_arr = np.full(payload.n_frames, task_index, dtype=np.int64)
    global_index_arr = np.arange(
        dataset_from_index, dataset_from_index + payload.n_frames, dtype=np.int64
    )

    # --- write data parquet (one episode per file in Sprint 2) ---
    data_dir = dst / "data" / f"chunk-{chunk_index:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"file-{file_index:03d}.parquet"
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index_arr, type=pa.int64()),
            "index": pa.array(global_index_arr, type=pa.int64()),
            "task_index": pa.array(task_index_arr, type=pa.int64()),
            "observation.state": _fixed_size_list(payload.state_22, 22),
            "action": _fixed_size_list(payload.action_22, 22),
        }
    )
    pq.write_table(table, data_path)
    cum_data_bytes += data_path.stat().st_size

    # --- encode video ---
    video_dir = dst / "videos" / HEAD_CAMERA_KEY / f"chunk-{chunk_index:03d}"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"file-{file_index:03d}.mp4"
    _reencode_video(payload.head_video, video_path, fps=FPS)
    cum_video_bytes += video_path.stat().st_size

    # --- write episode-metadata parquet row ---
    ep_meta_dir = dst / "meta" / "episodes" / f"chunk-{chunk_index:03d}"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_path = ep_meta_dir / f"file-{file_index:03d}.parquet"
    duration = payload.n_frames / float(FPS)
    s_state = _stats(payload.state_22)
    s_action = _stats(payload.action_22)
    row = {
        "episode_index": [episode_index],
        "tasks": [[payload.task_name]],
        "length": [int(payload.n_frames)],
        "meta/episodes/chunk_index": [chunk_index],
        "meta/episodes/file_index": [file_index],
        "data/chunk_index": [chunk_index],
        "data/file_index": [file_index],
        "dataset_from_index": [dataset_from_index],
        "dataset_to_index": [dataset_from_index + int(payload.n_frames)],
        f"videos/{HEAD_CAMERA_KEY}/chunk_index": [chunk_index],
        f"videos/{HEAD_CAMERA_KEY}/file_index": [file_index],
        f"videos/{HEAD_CAMERA_KEY}/from_timestamp": [0.0],
        f"videos/{HEAD_CAMERA_KEY}/to_timestamp": [duration],
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

    # --- update in-memory accumulators ---
    state_chunks.append(payload.state_22)
    action_chunks.append(payload.action_22)
    uuid_map_rows.append(
        {
            "episode_index": episode_index,
            "agibot_task": src_ep.task,
            "agibot_uuid": src_ep.uuid,
        }
    )

    # Move to next episode slot. Note: file_index advances per-episode in
    # Sprint 2's "one episode per file" policy. Reset cumulative counters too
    # since the next episode opens a fresh file.
    file_index += 1
    cum_data_bytes = 0
    cum_video_bytes = 0
    episode_index += 1
    return episode_index, chunk_index, file_index, cum_data_bytes, cum_video_bytes


# ---------------------------------------------------------------------------
# Resume support: scan existing dst → recover state.
# ---------------------------------------------------------------------------


def _load_resume_state(
    dst: Path,
) -> tuple[set[tuple[str, str]], int, int, int, int, int]:
    """Read meta/extra/uuid_map.parquet and return resume coordinates."""
    uuid_map = dst / "meta" / "extra" / "uuid_map.parquet"
    if not uuid_map.is_file():
        return set(), 0, 0, 0, 0, 0
    table = pq.read_table(uuid_map)
    done: set[tuple[str, str]] = set()
    max_ep = -1
    for task, uuid, ep in zip(
        table["agibot_task"].to_pylist(),
        table["agibot_uuid"].to_pylist(),
        table["episode_index"].to_pylist(),
        strict=True,
    ):
        done.add((task, uuid))
        if ep > max_ep:
            max_ep = ep
    next_episode_index = max_ep + 1
    # Find the highest chunk/file index used so far.
    next_chunk_index = 0
    next_file_index = 0
    for chunk_dir in sorted((dst / "data").glob("chunk-*")):
        ci = int(chunk_dir.name.split("-")[1])
        for f in sorted(chunk_dir.glob("file-*.parquet")):
            fi = int(f.stem.split("-")[1])
            if (ci, fi) > (next_chunk_index, next_file_index):
                next_chunk_index = ci
                next_file_index = fi
    # Advance past the last written file (since policy is one ep / file).
    next_file_index += 1
    return done, next_episode_index, next_chunk_index, next_file_index, 0, 0


def _load_existing_arrays(dst: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for chunk_dir in sorted((dst / "data").glob("chunk-*")):
        for f in sorted(chunk_dir.glob("file-*.parquet")):
            t = pq.read_table(f, columns=["observation.state", "action"])
            s = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
            a = np.array(t["action"].to_pylist(), dtype=np.float32)
            state_chunks.append(s)
            action_chunks.append(a)
    return state_chunks, action_chunks


def _load_task_index(dst: Path) -> dict[str, int]:
    p = dst / "meta" / "tasks.parquet"
    if not p.is_file():
        return {}
    t = pq.read_table(p)
    out: dict[str, int] = {}
    for name, idx in zip(t["task"].to_pylist(), t["task_index"].to_pylist(), strict=True):
        out[str(name)] = int(idx)
    return out


def _load_uuid_map(dst: Path) -> list[dict]:
    p = dst / "meta" / "extra" / "uuid_map.parquet"
    if not p.is_file():
        return []
    t = pq.read_table(p)
    return [
        {"episode_index": int(ei), "agibot_task": str(at), "agibot_uuid": str(au)}
        for ei, at, au in zip(
            t["episode_index"].to_pylist(),
            t["agibot_task"].to_pylist(),
            t["agibot_uuid"].to_pylist(),
            strict=True,
        )
    ]


# ---------------------------------------------------------------------------
# Aggregate writers (called once after all episodes committed).
# ---------------------------------------------------------------------------


def _finalize_aggregates(dst: Path) -> None:
    """Re-emit info.json/stats.json from on-disk parquets (no-op if dst empty)."""
    if not (dst / "data").exists():
        return
    state_chunks, action_chunks = _load_existing_arrays(dst)
    task_to_index = _load_task_index(dst)
    total_frames = sum(arr.shape[0] for arr in state_chunks)
    total_episodes = len(_load_uuid_map(dst))
    _write_stats_json_multi(dst, state_chunks=state_chunks, action_chunks=action_chunks)
    _write_info_json_multi(
        dst,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(task_to_index),
    )


def _write_tasks_parquet_multi(dst: Path, task_to_index: dict[str, int]) -> None:
    items = sorted(task_to_index.items(), key=lambda kv: kv[1])
    table = pa.table(
        {
            "task_index": pa.array([i for _, i in items], type=pa.int64()),
            "task": pa.array([n for n, _ in items], type=pa.string()),
        }
    )
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst / "meta" / "tasks.parquet")


def _write_uuid_map(dst: Path, rows: list[dict]) -> None:
    if not rows:
        return
    table = pa.table(
        {
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "agibot_task": pa.array([r["agibot_task"] for r in rows], type=pa.string()),
            "agibot_uuid": pa.array([r["agibot_uuid"] for r in rows], type=pa.string()),
        }
    )
    extra_dir = dst / "meta" / "extra"
    extra_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, extra_dir / "uuid_map.parquet")


def _write_stats_json_multi(
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
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "stats.json").write_text(json.dumps(payload, indent=2))


def _write_info_json_multi(
    dst: Path,
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": f"0:{int(total_frames)}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": list(JOINT_22),
            },
            "action": {
                "dtype": "float32",
                "shape": [22],
                "names": list(JOINT_22),
            },
            HEAD_CAMERA_KEY: {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"],
                "info": {
                    "video.fps": FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.height": 480,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.is_depth_map": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
# Single-episode internals (preserved from Sprint 1).
# ---------------------------------------------------------------------------


def _assert_digitalworld_sim(h5_path: Path) -> None:
    """Pre-flight: refuse real-robot AgiBot captures (Beta / Alpha) cleanly.

    DigitalWorld sim h5 has ``state/joint/position`` shape ``(N, 34)`` *and* a
    ``state/joint.attrs["name"]`` array of 34 joint names. Beta uses ``(N, 14)``
    with no ``name`` attr. We refuse instead of silently producing a broken
    LeRobot v3 dataset; real-robot ingest is on the v0.2 roadmap.
    """
    with h5py.File(h5_path, "r") as f:
        if "state/joint/position" not in f:
            raise ValueError(
                f"AgiBot HDF5 at {h5_path} is missing /state/joint/position; "
                "v0.1 supports DigitalWorld sim only — see docs/schema-agibot.md"
            )
        joint_dim = int(f["state/joint/position"].shape[1])
        joint_grp = f.get("state/joint")
        attrs_name = joint_grp.attrs.get("name") if joint_grp is not None else None
        attrs_missing = attrs_name is None
    if joint_dim != 34 or attrs_missing:
        raise ValueError(
            f"detected real-robot AgiBot capture (joint_dim={joint_dim}, "
            f"attrs.name={'present' if not attrs_missing else 'missing'}) — "
            "embodied-data v0.1 supports AgiBot DigitalWorld sim only. "
            "Real Alpha/Beta forward conversion is on the v0.2 roadmap; "
            "see docs/schema-agibot.md for detection rules."
        )


def _resolve_inputs(src: Path) -> tuple[Path, Path, Path]:
    h5_path = find_proprio_h5(src)
    task_info_path = src / "task_info.json"
    if h5_path is None:
        raise FileNotFoundError(
            f"expected proprio_states.h5 (or proprio_stats.h5 for Beta) under {src}"
        )
    if not task_info_path.is_file():
        raise FileNotFoundError(f"expected task_info.json under {src}")

    src_str = str(src)
    if "/meta_info/" not in src_str:
        raise ValueError(
            f"cannot resolve sibling video path: '/meta_info/' not in {src}; "
            "expected layout data/agibot_sample/meta_info/<task>/<uuid>/"
        )
    obs_dir = Path(src_str.replace("/meta_info/", "/observations/", 1))
    head_video = obs_dir / "video" / "head.mp4"
    if not head_video.is_file():
        raise FileNotFoundError(
            f"expected sibling video at {head_video}; "
            "AgiBot layout requires observations/ next to meta_info/"
        )
    return h5_path, task_info_path, head_video


def _read_state(h5_path: Path) -> tuple[np.ndarray, int, list[str]]:
    with h5py.File(h5_path, "r") as f:
        joint_names = list(f["state/joint"].attrs["name"])
        state_full = np.array(f["state/joint/position"], dtype=np.float32)

    missing = [n for n in JOINT_22 if n not in joint_names]
    if missing:
        raise ValueError(f"AgiBot HDF5 missing expected joint names: {missing}")

    indices = [joint_names.index(n) for n in JOINT_22]
    state_22 = state_full[:, indices].astype(np.float32, copy=False)
    n_frames = state_22.shape[0]
    if n_frames < 2:
        raise ValueError(f"need >=2 frames for first-diff action, got {n_frames}")
    return state_22, n_frames, joint_names


def _first_diff_action(state_22: np.ndarray) -> np.ndarray:
    diff = state_22[1:] - state_22[:-1]
    return np.concatenate([diff, diff[-1:]], axis=0).astype(np.float32)


def _read_task_name(task_info_path: Path) -> str:
    """Resolve task_name from a sim or Beta-style task_info JSON.

    DigitalWorld sim writes a single dict at ``<episode_dir>/task_info.json``.
    Beta writes a list of per-episode dicts at the *task root*
    (``task_info_<task_id>.json``); when that list shape leaks into the
    per-episode path we resolve it by matching ``episode_id`` against the
    parent directory name, falling back to entry ``[0]``.
    """
    with task_info_path.open() as f:
        info = json.load(f)
    if isinstance(info, list):
        if not info:
            raise ValueError(
                f"task_info JSON at {task_info_path} is an empty list; "
                "v0.1 only supports DigitalWorld sim (single-dict layout)"
            )
        episode_dir_name = task_info_path.parent.name
        # Try to match the parent dir (Beta uses int episode_id like 936938).
        try:
            target = int(episode_dir_name)
        except ValueError:
            target = None
        if target is not None:
            for entry in info:
                if isinstance(entry, dict) and entry.get("episode_id") == target:
                    return str(entry.get("task_name", "unknown"))
        # Fall back to first entry (still better than silently dropping the name).
        first = info[0]
        if isinstance(first, dict) and "task_name" in first:
            return str(first["task_name"])
        raise ValueError(
            f"task_info JSON at {task_info_path} is a list without a usable "
            "task_name; v0.1 only supports DigitalWorld sim"
        )
    if isinstance(info, dict):
        return str(info.get("task_name", "unknown"))
    raise ValueError(
        f"task_info JSON at {task_info_path} is neither dict nor list (got {type(info).__name__})"
    )


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


def _write_dataset_single(
    *,
    dst: Path,
    n_frames: int,
    state_22: np.ndarray,
    action_22: np.ndarray,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
    task_name: str,
    head_video: Path,
) -> None:
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "videos" / HEAD_CAMERA_KEY / "chunk-000").mkdir(parents=True, exist_ok=True)

    _write_info_json(dst, n_frames=n_frames)
    _write_tasks_parquet(dst, task_name=task_name)
    _write_data_parquet(
        dst,
        n_frames=n_frames,
        state_22=state_22,
        action_22=action_22,
        timestamps=timestamps,
        frame_index=frame_index,
    )
    _write_episodes_parquet(
        dst,
        n_frames=n_frames,
        task_name=task_name,
        state_22=state_22,
        action_22=action_22,
    )
    _write_stats_json(dst, state_22=state_22, action_22=action_22)
    _reencode_video(
        head_video,
        dst / "videos" / HEAD_CAMERA_KEY / "chunk-000" / "file-000.mp4",
        fps=FPS,
    )


def _write_info_json(dst: Path, *, n_frames: int) -> None:
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "total_episodes": 1,
        "total_frames": int(n_frames),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": f"0:{n_frames}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [22],
                "names": list(JOINT_22),
            },
            "action": {
                "dtype": "float32",
                "shape": [22],
                "names": list(JOINT_22),
            },
            HEAD_CAMERA_KEY: {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"],
                "info": {
                    "video.fps": FPS,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.height": 480,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.is_depth_map": False,
                },
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
    state_22: np.ndarray,
    action_22: np.ndarray,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
) -> None:
    episode_index = np.zeros(n_frames, dtype=np.int64)
    task_index = np.zeros(n_frames, dtype=np.int64)
    table = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(frame_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
            "observation.state": _fixed_size_list(state_22, 22),
            "action": _fixed_size_list(action_22, 22),
        }
    )
    pq.write_table(table, dst / "data" / "chunk-000" / "file-000.parquet")


def _write_episodes_parquet(
    dst: Path,
    *,
    n_frames: int,
    task_name: str,
    state_22: np.ndarray,
    action_22: np.ndarray,
) -> None:
    s_state = _stats(state_22)
    s_action = _stats(action_22)
    duration = n_frames / float(FPS)
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
        f"videos/{HEAD_CAMERA_KEY}/chunk_index": [0],
        f"videos/{HEAD_CAMERA_KEY}/file_index": [0],
        f"videos/{HEAD_CAMERA_KEY}/from_timestamp": [0.0],
        f"videos/{HEAD_CAMERA_KEY}/to_timestamp": [duration],
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
    table = pa.Table.from_pydict(row)
    pq.write_table(table, dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def _write_stats_json(dst: Path, *, state_22: np.ndarray, action_22: np.ndarray) -> None:
    payload = {
        "observation.state": _stats(state_22),
        "action": _stats(action_22),
    }
    (dst / "meta" / "stats.json").write_text(json.dumps(payload, indent=2))


def _reencode_video(src: Path, dst: Path, *, fps: int) -> None:
    # v3 footgun F1: never reuse upstream timestamps. Re-encode with monotonic
    # pts, small GOP, and B-frames disabled so DTS == PTS and the mp4 muxer
    # never sees out-of-order packets.
    time_base = Fraction(1, fps)
    in_container = av.open(str(src))
    in_stream = in_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    out_container = av.open(str(dst), mode="w")
    out_stream = out_container.add_stream("h264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "30", "g": "2", "bf": "0"}
    out_stream.codec_context.time_base = time_base

    try:
        for i, frame in enumerate(in_container.decode(in_stream)):
            new_frame = frame.reformat(format="yuv420p")
            new_frame.pts = i
            new_frame.time_base = time_base
            for packet in out_stream.encode(new_frame):
                out_container.mux(packet)
        for packet in out_stream.encode():
            out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()
