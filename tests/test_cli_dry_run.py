"""Tests for ``embodied-data convert --dry-run`` (Sprint 7 / Track B.1).

Goal: prove the flag does NOT write to disk and prints estimation output that
matches what the real convert would do, for the two paths v0.3.0 supports
(Beta single-episode + Beta batch). Reverse pair returns a clean exit-2 with
a "not supported yet" message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from embodied_data.cli import app

runner = CliRunner()

BETA_BATCH_ROOT = Path("data/agibot_beta_sample")
BETA_EPISODE = BETA_BATCH_ROOT / "675" / "936938"


@pytest.mark.skipif(
    not (BETA_EPISODE / "proprio_stats.h5").is_file(),
    reason="Beta single-episode fixture absent",
)
def test_convert_dry_run_does_not_create_dst(tmp_path: Path) -> None:
    dst = tmp_path / "dryrun_dst"
    assert not dst.exists()
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_EPISODE),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert not dst.exists(), f"dst {dst} was created during a dry-run"


@pytest.mark.skipif(
    not (BETA_EPISODE / "proprio_stats.h5").is_file(),
    reason="Beta single-episode fixture absent",
)
def test_convert_dry_run_prints_estimated_episodes(tmp_path: Path) -> None:
    dst = tmp_path / "dryrun_dst"
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_EPISODE),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "episodes" in out
    assert "1" in out
    assert "beta" in out.lower()


@pytest.mark.skipif(
    not (BETA_BATCH_ROOT / "task_info_675.json").is_file(),
    reason="Beta batch fixture absent",
)
def test_convert_dry_run_batch(tmp_path: Path) -> None:
    dst = tmp_path / "dryrun_batch_dst"
    assert not dst.exists()
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_BATCH_ROOT),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert not dst.exists(), f"dst {dst} was created during a batch dry-run"
    # Two episode dirs in the fixture tree (882736 + 936938).
    assert "2" in result.stdout
    assert "batch" in result.stdout


def test_convert_dry_run_lerobot_to_agibot_unsupported(tmp_path: Path) -> None:
    src = tmp_path / "fake_src"
    src.mkdir()
    dst = tmp_path / "fake_dst"
    result = runner.invoke(
        app,
        [
            "convert",
            str(src),
            str(dst),
            "--from",
            "lerobot-v3",
            "--to",
            "agibot",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "not supported" in result.stdout.lower() or "dry-run" in result.stdout.lower()
    assert not dst.exists()
