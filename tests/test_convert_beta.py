"""v0.2 integration tests for AgiBot Beta forward conversion.

WIP. The implementation is being built on ``feat/v0.2-real-beta-ingest``.
Tests in this file exercise the public contract:

- ``convert_agibot_beta_to_lerobot_v3(src, dst)`` lands a v3 dataset on disk
- ``observation.state`` has dim 20 per design §3
- timestamps are float32 seconds (ns→s conversion)
- action is first-difference of state, padded to length N
- the converted v3 passes ``embodied-data validate``

While the converter is a skeleton (raises ``NotImplementedError``), these
tests are marked ``xfail(strict=True)`` so a green CI immediately turns red
the moment the implementation starts producing output. Each test will get
flipped to a real assertion as the corresponding code lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied_data.convert.agibot_beta_to_lerobot import (
    JOINT_14_BETA,
    OBSERVATION_STATE_NAMES_20,
    convert_agibot_beta_to_lerobot_v3,
)

BETA_H5 = Path("data/agibot_beta_sample/675/936938/proprio_stats.h5")

needs_beta = pytest.mark.skipif(
    not BETA_H5.exists(),
    reason="beta fixture absent",
)


def test_module_constants():
    """Pin the v0.2 schema decisions: 14-joint best-guess + 20-dim state."""
    assert len(JOINT_14_BETA) == 14
    assert len(OBSERVATION_STATE_NAMES_20) == 20
    # First 14 are the joints, then effector pair, head pair, waist pair.
    assert OBSERVATION_STATE_NAMES_20[:14] == JOINT_14_BETA


def test_skeleton_raises_until_impl_lands():
    """Until the real converter ships, the skeleton must refuse loudly."""
    with pytest.raises(NotImplementedError, match="v0.2 WIP"):
        convert_agibot_beta_to_lerobot_v3(src=Path("/nope"), dst=Path("/nope"))


# ---------------------------------------------------------------------------
# Below: real-Beta integration tests. Marked xfail until impl lands. Each test
# documents what the implementation must produce. Strip the @pytest.mark.xfail
# decorator the moment the corresponding code path goes green.
# ---------------------------------------------------------------------------


@needs_beta
@pytest.mark.xfail(strict=True, reason="v0.2 WIP — converter skeleton only")
def test_beta_single_episode_smoke(tmp_path: Path):
    """Beta single-episode → produces meta/info.json + data parquet on disk."""
    convert_agibot_beta_to_lerobot_v3(src=BETA_H5.parent, dst=tmp_path / "v3")
    assert (tmp_path / "v3" / "meta" / "info.json").is_file()
    assert (tmp_path / "v3" / "data" / "chunk-000" / "file-000.parquet").is_file()


@needs_beta
@pytest.mark.xfail(strict=True, reason="v0.2 WIP")
def test_beta_state_dim_20(tmp_path: Path):
    """Output observation.state is 20-dim per design §3."""
    import json

    convert_agibot_beta_to_lerobot_v3(src=BETA_H5.parent, dst=tmp_path / "v3")
    info = json.loads((tmp_path / "v3" / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [20]


@needs_beta
@pytest.mark.xfail(strict=True, reason="v0.2 WIP")
def test_beta_timestamp_ns_to_s(tmp_path: Path):
    """Beta /timestamp int64 ns must convert to float32 seconds in parquet."""
    import pyarrow.parquet as pq

    convert_agibot_beta_to_lerobot_v3(src=BETA_H5.parent, dst=tmp_path / "v3")
    table = pq.read_table(tmp_path / "v3" / "data" / "chunk-000" / "file-000.parquet")
    ts_col = table["timestamp"].to_pylist()
    # frame_index/30 -> first frame is 0, second is 1/30 ≈ 0.0333 sec
    assert ts_col[0] == pytest.approx(0.0, abs=1e-6)
    assert ts_col[1] == pytest.approx(1 / 30, abs=1e-3)


@needs_beta
@pytest.mark.xfail(strict=True, reason="v0.2 WIP")
def test_beta_action_first_diff(tmp_path: Path):
    """action[i] = state[i+1] - state[i] for i in [0, N-2], pad last frame."""
    import numpy as np
    import pyarrow.parquet as pq

    convert_agibot_beta_to_lerobot_v3(src=BETA_H5.parent, dst=tmp_path / "v3")
    table = pq.read_table(tmp_path / "v3" / "data" / "chunk-000" / "file-000.parquet")
    state = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.array(table["action"].to_pylist(), dtype=np.float32)
    diff = state[1:] - state[:-1]
    np.testing.assert_allclose(action[:-1], diff, atol=1e-5)
    np.testing.assert_allclose(action[-1], action[-2], atol=1e-5)


@needs_beta
@pytest.mark.xfail(strict=True, reason="v0.2 WIP")
def test_beta_validate_passes(tmp_path: Path):
    """The Beta-converted v3 dataset passes embodied-data validate."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    convert_agibot_beta_to_lerobot_v3(src=BETA_H5.parent, dst=tmp_path / "v3")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(tmp_path / "v3")])
    assert result.exit_code == 0, result.output
