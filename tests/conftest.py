"""Shared fixtures for embodied-data tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Real one-episode AgiBot fixture under data/agibot_sample/.
SAMPLE_TASK = "digitaltwin_3"
SAMPLE_UUID = "000aa0b4-8fbe-432a-b6ae-559a7d7b3b96"
SAMPLE_META = Path(f"data/agibot_sample/meta_info/{SAMPLE_TASK}/{SAMPLE_UUID}")
SAMPLE_OBS = Path(f"data/agibot_sample/observations/{SAMPLE_TASK}/{SAMPLE_UUID}")


def _have_real_sample() -> bool:
    return (SAMPLE_META / "proprio_states.h5").is_file() and (
        SAMPLE_OBS / "video" / "head.mp4"
    ).is_file()


@pytest.fixture(scope="session")
def synthetic_agibot_root(tmp_path_factory) -> Path:
    """Build N synthetic AgiBot episodes by hard-link / copy from the real sample.

    Returns a path that can be passed as ``src`` to ``convert_agibot_batch``.
    Layout::

        <root>/meta_info/<SAMPLE_TASK>/ep_000/proprio_states.h5
        <root>/meta_info/<SAMPLE_TASK>/ep_000/task_info.json
        <root>/observations/<SAMPLE_TASK>/ep_000/video/head.mp4
        ... (10 total)

    Skips the test cleanly if the real fixture is missing.
    """
    if not _have_real_sample():
        pytest.skip("AgiBot sample fixture absent")

    root = tmp_path_factory.mktemp("agibot_synthetic")
    n_episodes = 10
    for i in range(n_episodes):
        uuid = f"ep_{i:03d}"
        meta_dir = root / "meta_info" / SAMPLE_TASK / uuid
        obs_dir = root / "observations" / SAMPLE_TASK / uuid / "video"
        meta_dir.mkdir(parents=True, exist_ok=True)
        obs_dir.mkdir(parents=True, exist_ok=True)
        # Symlink h5 + task_info to avoid copying 124MB × 10.
        for fname in ("proprio_states.h5", "task_info.json"):
            dst = meta_dir / fname
            if not dst.exists():
                dst.symlink_to((SAMPLE_META / fname).resolve())
        # Video must be a real path PyAV can open. Symlink works.
        head = obs_dir / "head.mp4"
        if not head.exists():
            head.symlink_to((SAMPLE_OBS / "video" / "head.mp4").resolve())

    yield root
    # tmp_path_factory cleans up automatically; nothing to do.
    shutil.rmtree  # noqa: B018  (silence unused-import warnings on some setups)
