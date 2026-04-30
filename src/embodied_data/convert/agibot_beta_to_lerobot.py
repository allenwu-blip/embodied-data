"""AgiBot Beta → LeRobot v3 forward converter (v0.2 WIP).

Skeleton only. The real conversion logic is being built incrementally on
``feat/v0.2-real-beta-ingest``. Until the implementation lands, calling this
function raises ``NotImplementedError`` pointing at the design doc.

Schema target: see ``docs/v0.2-real-beta-ingest.md``.
"""

from __future__ import annotations

from pathlib import Path

# Best-guess Beta joint name list (14 joints = 7-DoF dual arm). Documented as
# an assumption in docs/v0.2-real-beta-ingest.md §2; subject to revision once
# upstream URDF lands or a --joint-names override flag arrives in v0.2.1.
JOINT_14_BETA = [
    "arm_l_j1", "arm_l_j2", "arm_l_j3", "arm_l_j4",
    "arm_l_j5", "arm_l_j6", "arm_l_j7",
    "arm_r_j1", "arm_r_j2", "arm_r_j3", "arm_r_j4",
    "arm_r_j5", "arm_r_j6", "arm_r_j7",
]  # fmt: skip

# Final 20-dim observation.state vector composition for v0.2.
# 14 joint + 2 effector + 2 head + 2 waist; see design doc §3.
OBSERVATION_STATE_NAMES_20 = (
    JOINT_14_BETA
    + ["eff_l_width", "eff_r_width"]
    + ["head_yaw", "head_pitch"]
    + ["waist_yaw", "waist_pitch"]
)


def convert_agibot_beta_to_lerobot_v3(*, src: Path, dst: Path) -> None:
    """Convert one Beta episode dir to a LeRobot v3 dataset (v0.2 WIP)."""
    raise NotImplementedError(
        "agibot-beta -> lerobot-v3 is v0.2 WIP. "
        "See docs/v0.2-real-beta-ingest.md for design + scope. "
        "Until then, embodied-data v0.1.x refuses Beta input cleanly via "
        "convert_agibot_to_lerobot_v3's _assert_digitalworld_sim guard."
    )
