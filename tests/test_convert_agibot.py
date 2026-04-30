from __future__ import annotations

import json
from pathlib import Path

import av
import pyarrow.parquet as pq
import pytest

from embodied_data.convert.agibot_to_lerobot import (
    JOINT_22,
    convert_agibot_to_lerobot_v3,
)

SAMPLE_SRC = Path("data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")


def test_imports():
    assert callable(convert_agibot_to_lerobot_v3)
    assert len(JOINT_22) == 22


def test_missing_src_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_agibot_to_lerobot_v3(src=tmp_path / "nope", dst=tmp_path / "out")


@pytest.mark.skipif(not SAMPLE_SRC.exists(), reason="AgiBot sample fixture absent")
def test_convert_real_sample(tmp_path):
    dst = tmp_path / "lerobot_v3"
    convert_agibot_to_lerobot_v3(src=SAMPLE_SRC, dst=dst)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 30
    assert info["robot_type"] == "a2d"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 375
    assert info["total_tasks"] == 1
    assert info["features"]["observation.state"]["shape"] == [22]
    assert info["features"]["action"]["shape"] == [22]
    assert info["features"]["observation.images.top_head"]["dtype"] == "video"

    data_table = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    assert data_table.num_rows == 375
    cols = set(data_table.column_names)
    for required in {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.state",
        "action",
    }:
        assert required in cols

    state_row = data_table["observation.state"][0].as_py()
    action_row = data_table["action"][0].as_py()
    assert len(state_row) == 22
    assert len(action_row) == 22

    tasks_table = pq.read_table(dst / "meta" / "tasks.parquet")
    assert tasks_table.num_rows == 1
    task_str = tasks_table["task"][0].as_py()
    assert "place_objects" in task_str

    episodes_table = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes_table.num_rows == 1
    assert episodes_table["length"][0].as_py() == 375

    stats = json.loads((dst / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    assert stats["observation.state"]["count"] == [375]

    video_path = dst / "videos" / "observation.images.top_head" / "chunk-000" / "file-000.mp4"
    assert video_path.is_file()
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(stream))
        assert decoded == 375
        assert stream.average_rate == 30
    finally:
        container.close()
