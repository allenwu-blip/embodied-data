from __future__ import annotations

import json
import shutil
import uuid as _uuid
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from rich.console import Console

console = Console()

# Full 34-name AgiBot joint list (per docs/schema-agibot.md §3).
# Indices 22-33 are passive/mimic/support joints the forward converter drops.
# Reverse converter zero-fills them — round-trip cannot recover their values.
FULL_34 = [
    "joint_lift_body",            # 0
    "joint_body_pitch",           # 1
    "joint_head_yaw",             # 2
    "joint_head_pitch",           # 3
    "Joint1_l", "Joint1_r",       # 4, 5
    "Joint2_l", "Joint2_r",       # 6, 7
    "Joint3_l", "Joint3_r",       # 8, 9
    "Joint4_l", "Joint4_r",       # 10, 11
    "Joint5_l", "Joint5_r",       # 12, 13
    "Joint6_l", "Joint6_r",       # 14, 15
    "Joint7_l", "Joint7_r",       # 16, 17
    "left_Left_1_Joint",          # 18
    "left_Right_1_Joint",         # 19
    "right_Left_1_Joint",         # 20
    "right_Right_1_Joint",        # 21
    "left_Left_0_Joint",          # 22  dropped
    "left_Left_Support_Joint",    # 23  dropped
    "left_Right_0_Joint",         # 24  dropped
    "left_Right_Support_Joint",   # 25  dropped
    "right_Left_0_Joint",         # 26  dropped
    "right_Right_0_Joint",        # 27  dropped
    "Left_Left_RevoluteJoint",    # 28  dropped
    "Left_Right_RevoluteJoint",   # 29  dropped
    "right_Left_Support_Joint",   # 30  dropped
    "right_Right_Support_Joint",  # 31  dropped
    "right_Left_RevoluteJoint",   # 32  dropped
    "right_Right_RevoluteJoint",  # 33  dropped
]  # fmt: skip

# Map LeRobot v3 22-dim column index -> position in FULL_34.
# Forward order: head(2) + body(2) + arms(14) + effectors(4 right-then-left).
JOINT_22_TO_34 = [
    2,
    3,  # head_yaw, head_pitch
    0,
    1,  # lift_body, body_pitch
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,  # arms 1-7 left/right
    20,
    21,  # right_Left_1, right_Right_1
    18,
    19,  # left_Left_1, left_Right_1
]

FPS = 30
HEAD_CAMERA_KEY = "observation.images.top_head"


def convert_lerobot_v3_to_agibot(*, src: Path, dst: Path) -> None:
    """Reverse the v0.1 forward pipeline. Single-episode only.

    Output layout under ``dst``:
        meta_info/<task>/<uuid>/proprio_states.h5
        meta_info/<task>/<uuid>/task_info.json
        observations/<task>/<uuid>/video/head.mp4

    Lossy: the 12 dropped joints from the forward subselection are zero-filled.
    The video is byte-copied from the v3 mp4 (already h264 30fps post-forward).
    """
    src = Path(src)
    dst = Path(dst)

    info = _read_v3_info(src)
    state_22, n_frames = _read_v3_proprio(src)
    task_name = _read_v3_task(src)
    src_video = _resolve_v3_video(src)

    safe_task = _slugify(task_name)
    episode_uuid = str(_uuid.uuid4())
    meta_dir = dst / "meta_info" / safe_task / episode_uuid
    obs_dir = dst / "observations" / safe_task / episode_uuid / "video"
    meta_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    state_34 = _expand_22_to_34(state_22)
    # AgiBot raw action == state in the source sample (docs/schema-agibot.md §2.2),
    # so the round-trip writes state values back into action/joint/position.
    _write_agibot_h5(
        meta_dir / "proprio_states.h5",
        state_34=state_34,
        action_34=state_34,
        n_frames=n_frames,
    )
    _write_task_info_json(
        meta_dir / "task_info.json",
        task_name=task_name,
        episode_uuid=episode_uuid,
        safe_task=safe_task,
        n_frames=n_frames,
    )
    shutil.copy(src_video, obs_dir / "head.mp4")

    # TODO(v0.2): 7 other cameras (head_*_fisheye, back_*_fisheye, hand_left, hand_right)
    # TODO(v0.2): depth -> per-frame PNGs
    # TODO(v0.2): parameter.json calibration sidecar
    # TODO(v0.2): task_info.json.label_info.action_config from v3 subtasks.parquet
    # TODO(v0.2): multi-episode (split v3 batch into N AgiBot episode dirs)

    console.print(
        f"[green]done:[/green] {n_frames} frames → "
        f"{meta_dir.relative_to(dst)} (+ sibling video) | fps={info.get('fps')}"
    )


def _read_v3_info(src: Path) -> dict:
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"expected {info_path}; not a LeRobot v3 dataset?")
    info = json.loads(info_path.read_text())
    total_eps = info.get("total_episodes", 0)
    if total_eps != 1:
        raise NotImplementedError(
            f"multi-episode reverse conversion is v0.2; got total_episodes={total_eps}"
        )
    if info.get("fps") != FPS:
        raise NotImplementedError(
            f"v0.1 reverse only supports fps={FPS}; got fps={info.get('fps')}"
        )
    return info


def _read_v3_proprio(src: Path) -> tuple[np.ndarray, int]:
    parquet_path = src / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.is_file():
        candidates = sorted((src / "data").rglob("file-*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"no data parquet under {src / 'data'}")
        parquet_path = candidates[0]
    table = pq.read_table(parquet_path)
    state_22 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if state_22.ndim != 2 or state_22.shape[1] != 22:
        raise ValueError(f"expected observation.state of shape (N, 22); got {state_22.shape}")
    return state_22, int(state_22.shape[0])


def _read_v3_task(src: Path) -> str:
    tasks_path = src / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"expected {tasks_path}")
    table = pq.read_table(tasks_path)
    cols = table.column_names
    # F's preview discovered both layouts in the wild (column 'task' OR
    # pandas-index '__index_level_0__'). Accept either.
    for key in ("task", "__index_level_0__"):
        if key in cols:
            return str(table[key][0].as_py())
    raise KeyError(f"tasks.parquet has neither 'task' nor '__index_level_0__'; cols={cols}")


def _resolve_v3_video(src: Path) -> Path:
    primary = src / "videos" / HEAD_CAMERA_KEY / "chunk-000" / "file-000.mp4"
    if primary.is_file():
        return primary
    candidates = sorted((src / "videos" / HEAD_CAMERA_KEY).rglob("file-*.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"no mp4 found under {src / 'videos' / HEAD_CAMERA_KEY}; "
            "v3 dataset must have at least the top_head camera for v0.1 reverse"
        )
    return candidates[0]


def _slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return safe or "lerobot_v3_imported"


def _expand_22_to_34(state_22: np.ndarray) -> np.ndarray:
    n_frames = state_22.shape[0]
    state_34 = np.zeros((n_frames, 34), dtype=np.float32)
    for col_22, idx_34 in enumerate(JOINT_22_TO_34):
        state_34[:, idx_34] = state_22[:, col_22]
    return state_34


def _write_agibot_h5(
    path: Path,
    *,
    state_34: np.ndarray,
    action_34: np.ndarray,
    n_frames: int,
) -> None:
    name_dtype = h5py.string_dtype(encoding="utf-8")
    name_arr = np.array(FULL_34, dtype=name_dtype)
    with h5py.File(path, "w") as f:
        s_joint = f.create_group("state").create_group("joint")
        s_joint.create_dataset("position", data=state_34, dtype="float32")
        s_joint.attrs["name"] = name_arr

        a_joint = f.create_group("action").create_group("joint")
        a_joint.create_dataset("position", data=action_34, dtype="float32")
        a_joint.attrs["name"] = name_arr

        # Synthesized timestamps; AgiBot's upstream /timestamp is 60Hz-buggy
        # so we never propagate it.
        timestamps = (np.arange(n_frames) / float(FPS)).astype(np.float32)
        f.create_dataset("timestamp", data=timestamps, dtype="float32")


def _write_task_info_json(
    path: Path,
    *,
    task_name: str,
    episode_uuid: str,
    safe_task: str,
    n_frames: int,
) -> None:
    payload = {
        "task_name": task_name,
        "task_id": safe_task,
        "episode_id": episode_uuid,
        "label_info": {"action_config": []},
        "init_scene_text": "",
        "key_frame": [],
        "_imported_from": "lerobot_v3",
        "_n_frames": int(n_frames),
    }
    path.write_text(json.dumps(payload, indent=2))
