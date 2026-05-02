"""v0.3 (Sprint 7) — `embodied-data inspect <dir> --summary` integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied_data.convert.agibot_beta_to_lerobot import (
    convert_agibot_beta_to_lerobot_v3,
)

BETA_VIDEO_EPISODE = Path("data/agibot_beta_sample/675/882736")
BETA_LEGACY_EPISODE = Path("data/agibot_beta_sample/675/936938")

needs_video_fixture = pytest.mark.skipif(
    not (BETA_VIDEO_EPISODE / "videos" / "head_color.mp4").is_file(),
    reason="Beta video fixture absent (run scripts/fetch_beta_video_fixture.py)",
)
needs_legacy_fixture = pytest.mark.skipif(
    not BETA_LEGACY_EPISODE.is_dir(),
    reason="Beta legacy fixture absent",
)


@needs_video_fixture
def test_summary_with_video_dataset(tmp_path: Path):
    """Summary on a Beta-converted dataset with head_color video reports the
    overall PASS status and lists all expected fields."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(dst), "--summary"])
    assert result.exit_code == 0, result.output
    out = result.output
    # Header line
    assert "Dataset summary" in out
    # Overview fields
    for label in (
        "Robot type",
        "FPS",
        "Episodes",
        "Frames",
        "Duration",
        "State dim",
        "Action dim",
        "Cameras",
        "Disk size",
    ):
        assert label in out, f"missing label '{label}' in:\n{out}"
    # Robot type, fps, episodes, frames pulled straight from info.json
    assert "agibot-beta" in out
    assert "879" in out  # frame count
    # Camera detail
    assert "head_color" in out
    assert "640x480" in out
    assert "h264" in out
    # Mini-validate output present
    assert "Validation checks" in out
    assert "Overall: " in out
    assert "PASS" in out


@needs_legacy_fixture
def test_summary_with_proprio_only_dataset(tmp_path: Path):
    """Summary on a proprio-only Beta dataset reports zero cameras and still
    reaches an Overall verdict (frame-video alignment SKIPs cleanly)."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_LEGACY_EPISODE, dst=dst)

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(dst), "--summary"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Dataset summary" in out
    # No video features → no head_color row in the cameras subtable
    assert "head_color" not in out
    # frame-video alignment should SKIP cleanly (info.features has no video dtype)
    assert "SKIP" in out
    # Still reaches an overall verdict
    assert "Overall: PASS" in out

    # Dual check via JSON output for the cameras=[] invariant — terminal-
    # width-independent (Rich tables wrap unpredictably under CliRunner).
    import json

    json_result = runner.invoke(app, ["--json", "inspect", str(dst), "--summary"])
    payload = json.loads(json_result.output)
    assert payload["cameras"] == []
    assert payload["total_frames"] == 1090


def test_summary_rejects_nonexistent_path(tmp_path: Path):
    """Summary on a path that doesn't exist exits with the path-error code."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    runner = CliRunner()
    bogus = tmp_path / "does_not_exist"
    result = runner.invoke(app, ["inspect", str(bogus), "--summary"])
    assert result.exit_code != 0


def test_summary_rejects_file_path(tmp_path: Path):
    """Summary on a single file (not a dir) exits non-zero with helpful suggestion."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    f = tmp_path / "stray.parquet"
    f.write_bytes(b"")
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(f), "--summary"])
    assert result.exit_code != 0


def test_summary_rejects_directory_without_meta_info(tmp_path: Path):
    """Summary on a dir that's not a v3 dataset (missing meta/info.json) errors."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    fake_root = tmp_path / "not_a_v3_dataset"
    fake_root.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(fake_root), "--summary"])
    assert result.exit_code != 0


@needs_video_fixture
def test_summary_json_output(tmp_path: Path):
    """`--json inspect <dir> --summary` emits a JSON dict with all expected keys."""
    import json

    from typer.testing import CliRunner

    from embodied_data.cli import app

    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)

    runner = CliRunner()
    result = runner.invoke(app, ["--json", "inspect", str(dst), "--summary"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    expected_keys = {
        "path",
        "fps",
        "robot_type",
        "total_episodes",
        "total_frames",
        "duration_seconds",
        "state_dim",
        "action_dim",
        "cameras",
        "disk_bytes",
        "validate_results",
        "overall_status",
    }
    assert expected_keys.issubset(payload.keys()), payload.keys()
    assert payload["overall_status"] == "PASS"
    assert payload["total_frames"] == 879
    assert payload["state_dim"] == 20
    assert len(payload["cameras"]) == 1
    assert payload["cameras"][0]["key"] == "observation.images.head_color"


def test_human_bytes_formats():
    """Unit check on the disk-size humanizer."""
    from embodied_data.inspect import _human_bytes

    assert _human_bytes(0) == "0 B"
    assert _human_bytes(1023) == "1023 B"
    assert _human_bytes(1024).endswith("KB")
    assert _human_bytes(1024 * 1024).endswith("MB")
    assert _human_bytes(5_000_000).startswith("4.")  # 4.7 MB
