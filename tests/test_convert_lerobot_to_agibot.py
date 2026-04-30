from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from embodied_data.convert.agibot_to_lerobot import convert_agibot_to_lerobot_v3
from embodied_data.convert.lerobot_to_agibot import (
    FULL_34,
    JOINT_22_TO_34,
    convert_lerobot_v3_to_agibot,
)

SAMPLE_SRC = Path("data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")


def test_imports_and_constants():
    assert callable(convert_lerobot_v3_to_agibot)
    assert len(FULL_34) == 34
    assert len(JOINT_22_TO_34) == 22
    assert max(JOINT_22_TO_34) < 34
    # 22 distinct destination indices, all in [0, 22) range = preserved joints
    assert sorted(JOINT_22_TO_34) == list(range(22))


def test_missing_v3_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_lerobot_v3_to_agibot(src=tmp_path / "nope", dst=tmp_path / "out")


def test_multi_episode_v3_rejected(tmp_path):
    # Build a fake v3 with total_episodes=2 → expect NotImplementedError
    fake = tmp_path / "fake_v3"
    (fake / "meta").mkdir(parents=True)
    (fake / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 30, "total_episodes": 2})
    )
    with pytest.raises(NotImplementedError, match="multi-episode"):
        convert_lerobot_v3_to_agibot(src=fake, dst=tmp_path / "out")


def test_wrong_fps_v3_rejected(tmp_path):
    fake = tmp_path / "fake_v3"
    (fake / "meta").mkdir(parents=True)
    (fake / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 50, "total_episodes": 1})
    )
    with pytest.raises(NotImplementedError, match="fps"):
        convert_lerobot_v3_to_agibot(src=fake, dst=tmp_path / "out")


@pytest.mark.skipif(not SAMPLE_SRC.exists(), reason="AgiBot sample fixture absent")
def test_reverse_writes_expected_layout(tmp_path):
    forward_dst = tmp_path / "v3"
    convert_agibot_to_lerobot_v3(src=SAMPLE_SRC, dst=forward_dst)

    reverse_dst = tmp_path / "agibot_back"
    convert_lerobot_v3_to_agibot(src=forward_dst, dst=reverse_dst)

    h5_paths = list((reverse_dst / "meta_info").rglob("proprio_states.h5"))
    assert len(h5_paths) == 1
    h5_path = h5_paths[0]

    task_info_path = h5_path.parent / "task_info.json"
    assert task_info_path.is_file()

    parts = h5_path.parent.relative_to(reverse_dst / "meta_info").parts
    assert len(parts) == 2  # <task>/<uuid>
    safe_task, episode_uuid = parts

    video_path = reverse_dst / "observations" / safe_task / episode_uuid / "video" / "head.mp4"
    assert video_path.is_file()
    assert video_path.stat().st_size > 0

    info = json.loads(task_info_path.read_text())
    assert info["task_name"]
    assert info["episode_id"] == episode_uuid
    assert info["_imported_from"] == "lerobot_v3"

    with h5py.File(h5_path) as f:
        assert f["state/joint/position"].shape == (375, 34)
        assert f["state/joint/position"].dtype == np.float32
        names = list(f["state/joint"].attrs["name"])
        assert names == FULL_34
        assert f["timestamp"].shape == (375,)
        # action == state per AgiBot convention
        np.testing.assert_array_equal(
            f["state/joint/position"][:],
            f["action/joint/position"][:],
        )
