from pathlib import Path

import pytest
from typer.testing import CliRunner

from embodied_data.cli import app
from embodied_data.validate import _detect, run_validate
from embodied_data.validate.agibot import (
    check_action_dim,
    check_alignment,
    check_fps,
    check_timestamp,
)

runner = CliRunner()

_EP = "digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96"
AGIBOT_SAMPLE = Path(f"data/agibot_sample/meta_info/{_EP}")
AGIBOT_VIDEO_DIR = Path(f"data/agibot_sample/observations/{_EP}/video")


def test_nonexistent_path_exits_2(tmp_path):
    missing = tmp_path / "no-such-dir"
    result = runner.invoke(app, ["validate", str(missing)])
    assert result.exit_code == 2


def test_unknown_format_exits_2(tmp_path):
    # tmp_path exists but has no meta/info.json or proprio_states.h5
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 2


def test_detect_lerobot_v3(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    assert _detect(tmp_path) == "lerobot-v3"


def test_detect_agibot(tmp_path):
    (tmp_path / "proprio_states.h5").write_bytes(b"")
    assert _detect(tmp_path) == "agibot"


def test_detect_unknown(tmp_path):
    assert _detect(tmp_path) == "unknown"


@pytest.mark.skipif(not AGIBOT_SAMPLE.exists(), reason="agibot sample fixture absent")
def test_agibot_sample_warn_on_timestamp_no_fail():
    # Build a synthetic dir layout matching path.glob lookup
    from embodied_data.validate import agibot

    h5 = agibot._find_h5(AGIBOT_SAMPLE)
    videos = sorted(AGIBOT_VIDEO_DIR.glob("*.mp4"))
    assert h5 is not None
    assert len(videos) == 8

    fps = check_fps(videos)
    ts = check_timestamp(h5, videos)
    dim = check_action_dim(h5)
    align = check_alignment(h5, videos)

    assert fps.status == "PASS", fps
    # timestamp should be WARN (60Hz timestamp vs 30Hz video) — not FAIL
    assert ts.status == "WARN", ts
    assert "60" in ts.detail or "implies" in ts.detail
    assert dim.status == "PASS", dim
    assert align.status == "PASS", align


@pytest.mark.skipif(not AGIBOT_SAMPLE.exists(), reason="agibot sample fixture absent")
def test_agibot_combined_dir(tmp_path):
    """Compose a dir with both proprio_states.h5 and a video/ subdir, run validate."""
    import os

    combined = tmp_path / "episode"
    combined.mkdir()
    os.symlink(
        (AGIBOT_SAMPLE / "proprio_states.h5").resolve(),
        combined / "proprio_states.h5",
    )
    os.symlink(AGIBOT_VIDEO_DIR.resolve(), combined / "video")

    result = runner.invoke(app, ["validate", str(combined)])
    # WARN does not raise, exit 0
    assert result.exit_code == 0, result.stdout
    assert "agibot" in result.stdout
    assert "WARN" in result.stdout or "warning" in result.stdout.lower()


def test_run_validate_direct_unknown(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_validate(path=tmp_path, fmt="auto")
    assert exc.value.code == 2
