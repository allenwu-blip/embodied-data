"""Unit + integration tests for ``embodied_data._agibot_paths`` (M2 dispatcher).

Covers ``detect_agibot_variant`` against:
- sim DigitalWorld fixture (``data/agibot_sample/``)
- real Beta fixture (``data/agibot_beta_sample/675/936938``)
- Alpha path-name hint (synthetic tmp dir with ``alpha`` in path)
- unknown layout (synthetic empty / wrong-content tmp dir)
- sim batch root heuristic (rglob discovery)

Plus end-to-end CLI dispatch smoke tests for each variant.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from embodied_data._agibot_paths import (
    detect_agibot_variant,
    schema_summary,
)
from embodied_data.cli import app

SIM_EP_DIR = Path("data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")
SIM_BATCH_ROOT = Path("data/agibot_sample/meta_info")
BETA_EP_DIR = Path("data/agibot_beta_sample/675/936938")
BETA_H5 = BETA_EP_DIR / "proprio_stats.h5"

needs_sim = pytest.mark.skipif(not SIM_EP_DIR.exists(), reason="sim fixture absent")
needs_beta = pytest.mark.skipif(not BETA_EP_DIR.exists(), reason="beta fixture absent")

runner = CliRunner()


# ---------------------------------------------------------------------------
# detect_agibot_variant — unit
# ---------------------------------------------------------------------------


@needs_sim
def test_detect_sim_single_episode():
    assert detect_agibot_variant(SIM_EP_DIR) == "digitalworld"


@needs_sim
def test_detect_sim_batch_root():
    """Sim batch root has no h5 directly but rglob finds proprio_states.h5."""
    assert detect_agibot_variant(SIM_BATCH_ROOT) == "digitalworld"


@needs_beta
def test_detect_beta_single_episode():
    assert detect_agibot_variant(BETA_EP_DIR) == "beta"


@needs_beta
def test_detect_beta_task_root():
    """As of M3, Beta task-dataset root (task_info_*.json sibling + Beta-named
    h5 in subtree) is recognised as 'beta' so the dispatcher can auto-batch."""
    beta_root = BETA_EP_DIR.parent.parent  # data/agibot_beta_sample/
    assert detect_agibot_variant(beta_root) == "beta"


def test_detect_alpha_path_hint(tmp_path: Path):
    """Path component containing 'alpha' (case-insensitive) maps to the Alpha
    stub. This is a heuristic — Alpha and Beta share schemas per upstream README,
    so on-disk content alone cannot distinguish them."""
    alpha_dir = tmp_path / "AgiBotWorld-Alpha" / "task_X" / "ep_0"
    alpha_dir.mkdir(parents=True)
    assert detect_agibot_variant(alpha_dir) == "alpha"


def test_detect_unknown_empty_dir(tmp_path: Path):
    """Empty dir → unknown."""
    assert detect_agibot_variant(tmp_path) == "unknown"


def test_detect_unknown_nonexistent(tmp_path: Path):
    assert detect_agibot_variant(tmp_path / "does_not_exist") == "unknown"


def test_detect_unknown_h5_with_wrong_shape(tmp_path: Path):
    """An h5 with proprio filename but wrong joint shape → unknown (not Beta or sim)."""
    bogus = tmp_path / "proprio_states.h5"
    with h5py.File(bogus, "w") as f:
        f.create_dataset("state/joint/position", data=np.zeros((10, 7), dtype=np.float32))
    # state/joint group exists but has no name attr
    assert detect_agibot_variant(tmp_path) == "unknown"


# ---------------------------------------------------------------------------
# schema_summary — unit
# ---------------------------------------------------------------------------


@needs_beta
def test_schema_summary_beta():
    s = schema_summary(BETA_EP_DIR)
    assert "proprio_stats.h5" in s
    assert "(1090, 14)" in s
    assert "missing" in s  # attrs.name


@needs_sim
def test_schema_summary_sim():
    s = schema_summary(SIM_EP_DIR)
    assert "proprio_states.h5" in s
    assert "(375, 34)" in s
    assert "present" in s


def test_schema_summary_nonexistent(tmp_path: Path):
    s = schema_summary(tmp_path / "nope")
    assert "does not exist" in s


# ---------------------------------------------------------------------------
# CLI dispatch — integration
# ---------------------------------------------------------------------------


@needs_beta
def test_cli_alpha_path_routes_through_beta(tmp_path: Path):
    """As of M3 + B.1 empirical verification, Alpha schemas are equivalent to
    Beta. The dispatcher routes ``variant=='alpha'`` paths through the Beta
    converter with a one-line console note, instead of the v0.2-pre stub error.

    Pin the routing by feeding the Beta fixture under an Alpha-named path."""
    alpha_dir = tmp_path / "AgiBotWorld-Alpha" / "675" / "936938"
    alpha_dir.mkdir(parents=True)
    (alpha_dir / "proprio_stats.h5").symlink_to(BETA_H5.resolve())

    result = runner.invoke(
        app,
        [
            "convert",
            str(alpha_dir),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 0, result.output
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["robot_type"] == "agibot-beta"  # routed via Beta converter
    assert info["features"]["observation.state"]["shape"] == [20]
    assert "alpha" in result.output.lower()  # the routing note must be visible


def test_cli_unknown_variant_error(tmp_path: Path):
    """`embodied-data convert <empty-dir>` exits 2 with schema summary."""
    empty = tmp_path / "weird_layout"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "convert",
            str(empty),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "could not identify" in result.output.lower() or "unknown" in result.output.lower()
    assert "schema summary" in result.output.lower()


@needs_sim
def test_cli_sim_path_unaffected_by_m2(tmp_path: Path):
    """Regression: sim DigitalWorld single-episode still routes through the sim
    converter and writes a 22-dim state v3 dataset (M2 routing must not break sim)."""
    result = runner.invoke(
        app,
        [
            "convert",
            str(SIM_EP_DIR),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 0, result.output
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["robot_type"] == "a2d"  # sim's hardcoded robot_type literal
    assert info["features"]["observation.state"]["shape"] == [22]


@needs_beta
def test_cli_beta_path_routes_to_beta_converter(tmp_path: Path):
    """End-to-end: `embodied-data convert <beta-ep-dir>` produces a 20-dim Beta v3 dataset."""
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_EP_DIR),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 0, result.output
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["robot_type"] == "agibot-beta"
    assert info["features"]["observation.state"]["shape"] == [20]


@needs_beta
def test_cli_beta_single_episode_dir_refuses_batch_flags(tmp_path: Path):
    """As of M3 the dispatcher allows Beta batch — but only if src is a task
    root, not a single-episode dir. Passing --max-episodes against a single
    episode dir exits 2 with a clear hint about the layout mismatch."""
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_EP_DIR),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--max-episodes",
            "1",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "single-episode" in result.output.lower() or "task root" in result.output.lower()
