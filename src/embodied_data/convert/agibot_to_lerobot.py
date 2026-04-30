from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

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


def convert_agibot_to_lerobot_v3(*, src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)

    h5_path, task_info_path, head_video = _resolve_inputs(src)

    state_22, n_frames, joint_names = _read_state(h5_path)
    action_22 = _first_diff_action(state_22)
    task_name = _read_task_name(task_info_path)

    timestamps = (np.arange(n_frames) / float(FPS)).astype(np.float32)
    frame_index = np.arange(n_frames, dtype=np.int64)

    _write_dataset(
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


def _resolve_inputs(src: Path) -> tuple[Path, Path, Path]:
    h5_path = src / "proprio_states.h5"
    task_info_path = src / "task_info.json"
    if not h5_path.is_file():
        raise FileNotFoundError(f"expected proprio_states.h5 under {src}")
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
    with task_info_path.open() as f:
        info = json.load(f)
    return str(info.get("task_name", "unknown"))


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


def _write_dataset(
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

    # TODO(v0.1): add 7 other cameras (fisheyes, hand_left, hand_right)
    # TODO(v0.1): add depth as observation.images.cam_top_depth (PNG-bytes image feature)
    # TODO(v0.1): multi-episode batches with chunk rollover


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
