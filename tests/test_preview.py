from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from embodied_data.cli import app
from embodied_data.preview import detect_format

runner = CliRunner()

AGIBOT_SAMPLE = Path(
    "data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96"
)


def test_preview_imports():
    from embodied_data.preview import run_preview  # noqa: F401


def test_preview_unknown_path_exits_2(tmp_path: Path):
    result = runner.invoke(app, ["preview", str(tmp_path / "nope")])
    assert result.exit_code == 2


def test_preview_empty_dir_exits_2(tmp_path: Path):
    result = runner.invoke(app, ["preview", str(tmp_path)])
    assert result.exit_code == 2


def test_detect_lerobot_v3(tmp_path: Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    assert detect_format(tmp_path) == "lerobot-v3"


def test_detect_agibot_via_h5(tmp_path: Path):
    (tmp_path / "proprio_states.h5").write_bytes(b"")
    assert detect_format(tmp_path) == "agibot"


def test_detect_agibot_via_sibling_scripts(tmp_path: Path):
    # Mimic <root>/scripts/convert_to_lerobot.py + <root>/meta_info/<task>/<uuid>/
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "convert_to_lerobot.py").write_text("# stub")
    episode = root / "meta_info" / "task" / "uuid"
    episode.mkdir(parents=True)
    assert detect_format(episode) == "agibot"


def test_lerobot_v3_corrupt_info_exits_2(tmp_path: Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("not-json{")
    result = runner.invoke(app, ["preview", str(tmp_path)])
    assert result.exit_code == 2


def test_lerobot_v3_minimal_renders(tmp_path: Path):
    """Synthetic v3 dataset with no episodes parquet — should still render features."""
    (tmp_path / "meta").mkdir()
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": "test_bot",
        "total_episodes": 0,
        "total_frames": 0,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            "observation.images.top": {"dtype": "video", "shape": [3, 480, 640]},
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    result = runner.invoke(app, ["preview", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "lerobot-v3" in result.output
    assert "test_bot" in result.output
    assert "observation.images.top" in result.output


@pytest.mark.skipif(not AGIBOT_SAMPLE.exists(), reason="agibot sample fixture not present")
def test_agibot_sample_preview_renders():
    result = runner.invoke(app, ["preview", str(AGIBOT_SAMPLE)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "agibot" in out
    assert "375" in out
    assert "30" in out  # fps
    assert "22" in out and "34" in out  # state dim subselected + raw
    assert "place_objects_into_handbag" in out
    # All 8 cameras should be listed.
    for cam in ("head", "hand_left", "hand_right", "back_left_fisheye"):
        assert cam in out


@pytest.mark.skipif(not AGIBOT_SAMPLE.exists(), reason="agibot sample fixture not present")
def test_agibot_sample_n_zero_falls_back_to_default():
    # n=0 must not crash; treated as default 10.
    result = runner.invoke(app, ["preview", str(AGIBOT_SAMPLE), "-n", "0"])
    assert result.exit_code == 0, result.output
    assert "375" in result.output
