from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from embodied_data.convert.agibot_to_lerobot import (
    JOINT_22,
    convert_agibot_to_lerobot_v3,
)
from embodied_data.convert.lerobot_to_agibot import (
    FULL_34,
    convert_lerobot_v3_to_agibot,
)

SAMPLE_SRC = Path("data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")
DROPPED_NAMES = [n for n in FULL_34 if n not in JOINT_22]


@pytest.mark.skipif(not SAMPLE_SRC.exists(), reason="AgiBot sample fixture absent")
def test_round_trip_preserves_22_joints_and_timestamp(tmp_path):
    forward_dst = tmp_path / "v3"
    convert_agibot_to_lerobot_v3(src=SAMPLE_SRC, dst=forward_dst)

    reverse_dst = tmp_path / "agibot_back"
    convert_lerobot_v3_to_agibot(src=forward_dst, dst=reverse_dst)

    rt_h5_paths = list((reverse_dst / "meta_info").rglob("proprio_states.h5"))
    assert len(rt_h5_paths) == 1
    rt_h5 = rt_h5_paths[0]

    with (
        h5py.File(SAMPLE_SRC / "proprio_states.h5", "r") as orig,
        h5py.File(rt_h5, "r") as rt,
    ):
        orig_state = np.asarray(orig["state/joint/position"], dtype=np.float32)
        orig_names = [str(n) for n in orig["state/joint"].attrs["name"]]
        rt_state = np.asarray(rt["state/joint/position"], dtype=np.float32)
        rt_names = [str(n) for n in rt["state/joint"].attrs["name"]]
        rt_ts = np.asarray(rt["timestamp"], dtype=np.float32)

    assert orig_state.shape == rt_state.shape == (375, 34)

    max_diff = 0.0
    for name in JOINT_22:
        oi = orig_names.index(name)
        ri = rt_names.index(name)
        diff = float(np.max(np.abs(orig_state[:, oi] - rt_state[:, ri])))
        max_diff = max(max_diff, diff)
        np.testing.assert_allclose(
            orig_state[:, oi],
            rt_state[:, ri],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"joint {name} diverged in round-trip",
        )

    print(f"\nround-trip max abs diff across 22 preserved joints = {max_diff:.3e}")

    n_frames = orig_state.shape[0]
    expected_ts = (np.arange(n_frames) / 30.0).astype(np.float32)
    np.testing.assert_array_equal(rt_ts, expected_ts)

    for d_name in DROPPED_NAMES:
        ri = rt_names.index(d_name)
        assert np.all(rt_state[:, ri] == 0), (
            f"dropped joint {d_name} should be zero-filled in round-trip output"
        )


@pytest.mark.skipif(not SAMPLE_SRC.exists(), reason="AgiBot sample fixture absent")
def test_round_trip_video_byte_size_nonzero(tmp_path):
    forward_dst = tmp_path / "v3"
    convert_agibot_to_lerobot_v3(src=SAMPLE_SRC, dst=forward_dst)

    reverse_dst = tmp_path / "agibot_back"
    convert_lerobot_v3_to_agibot(src=forward_dst, dst=reverse_dst)

    rt_videos = list((reverse_dst / "observations").rglob("head.mp4"))
    assert len(rt_videos) == 1
    assert rt_videos[0].stat().st_size > 0
