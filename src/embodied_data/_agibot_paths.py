"""Filename-resolution + variant-detection helpers shared across CLI commands.

Real AgiBot Beta captures use ``proprio_stats.h5`` (no second ``s`` in
"states") while the published DigitalWorld sim uses ``proprio_states.h5``.
v0.1.0 hard-coded the sim filename in ~7 call sites and silently failed on
real-Beta data. This module centralises the lookup + variant detection so the
``convert`` dispatcher routes sim, Beta, and (future) Alpha inputs to their
respective converters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py

# Glob pattern matching both filenames in a single ``Path.glob`` / ``rglob``
# call. Order in the character class is irrelevant; both letters are accepted.
PROPRIO_GLOB = "proprio_stat[se]*.h5"

AgibotVariant = Literal["digitalworld", "beta", "alpha", "unknown"]


def find_proprio_h5(path: Path) -> Path | None:
    """Return the proprio h5 file directly inside ``path``, if any.

    Searches only the immediate directory (no recursion). Prefers
    ``proprio_states.h5`` (sim DigitalWorld) over ``proprio_stats.h5`` (Beta)
    when both happen to exist, so legacy sim users keep their behaviour.
    """
    if not path.is_dir():
        return None
    sim = path / "proprio_states.h5"
    if sim.is_file():
        return sim
    beta = path / "proprio_stats.h5"
    if beta.is_file():
        return beta
    return None


def iter_proprio_h5(root: Path) -> list[Path]:
    """Recursively find every proprio h5 (sim or Beta naming) under ``root``."""
    if not root.exists():
        return []
    return [p for p in root.rglob(PROPRIO_GLOB) if p.is_file()]


def detect_agibot_variant(path: Path) -> AgibotVariant:
    """Identify which AgiBot variant ``path`` contains.

    Returns one of ``digitalworld | beta | alpha | unknown``.

    Detection signals (in order of precedence):

    1. **Path naming hint for Alpha** — Alpha and Beta ship structurally
       identical schemas per the upstream README, so on-disk content alone
       cannot distinguish them. If any directory component contains "alpha"
       (case-insensitive), the path is treated as Alpha. The router uses this
       to surface a friendly "Alpha is a v0.2 follow-up milestone" error
       rather than silently routing to Beta.
    2. **proprio_states.h5 (sim filename) + 34-dim joint + name attr**
       → ``digitalworld``.
    3. **proprio_stats.h5 (Beta filename) + 14-dim joint + missing name attr**
       → ``beta``.
    4. **No h5 in immediate dir but sim batch layout** (subtree contains
       ``meta_info/<task>/<uuid>/proprio_states.h5``) → ``digitalworld`` (the
       sim batch path; Beta batch is a future milestone).
    5. **Otherwise** → ``unknown``.
    """
    path = Path(path)
    if not path.exists():
        return "unknown"

    # Step 1: Alpha path-name hint.
    if any("alpha" in part.lower() for part in path.parts):
        return "alpha"

    # Steps 2-3: schema detection on the immediate-directory h5.
    h5_path = find_proprio_h5(path) if path.is_dir() else None
    if h5_path is not None:
        try:
            with h5py.File(h5_path, "r") as f:
                joint = f.get("state/joint/position")
                if joint is None or joint.ndim != 2:
                    return "unknown"
                joint_dim = int(joint.shape[1])
                joint_grp = f.get("state/joint")
                attrs_name = joint_grp.attrs.get("name") if joint_grp is not None else None
        except (OSError, KeyError, ValueError):
            return "unknown"
        if joint_dim == 34 and attrs_name is not None:
            return "digitalworld"
        if joint_dim == 14 and attrs_name is None:
            return "beta"
        return "unknown"

    # Step 4: Beta batch root heuristic — has task_info_*.json sibling and at
    # least one Beta-named h5 in the subtree. Active as of M3.
    if path.is_dir() and any(path.glob("task_info_*.json")):
        for hit in path.rglob("proprio_stats.h5"):
            if hit.is_file():
                return "beta"

    # Step 5: sim batch root heuristic — any sim-named h5 anywhere in subtree.
    if path.is_dir():
        for hit in path.rglob("proprio_states.h5"):
            if hit.is_file():
                return "digitalworld"

    return "unknown"


def schema_summary(path: Path) -> str:
    """One-line schema summary for use in 'unknown variant' error messages."""
    if not path.exists():
        return f"path {path} does not exist"
    if not path.is_dir():
        return f"path {path} is not a directory"
    h5 = find_proprio_h5(path)
    if h5 is None:
        files = sorted(p.name for p in path.iterdir())[:8]
        return f"no proprio_state[se]*.h5 in {path}; first 8 entries: {files}"
    try:
        with h5py.File(h5, "r") as f:
            j = f.get("state/joint/position")
            shape = tuple(j.shape) if j is not None else "missing"
            attrs = (
                "present"
                if (f.get("state/joint") and f["state/joint"].attrs.get("name") is not None)
                else "missing"
            )
        return f"h5={h5.name}, state/joint/position shape={shape}, state/joint.attrs.name={attrs}"
    except OSError as exc:
        return f"h5={h5.name}, read error: {exc}"


__all__ = [
    "AgibotVariant",
    "PROPRIO_GLOB",
    "detect_agibot_variant",
    "find_proprio_h5",
    "iter_proprio_h5",
    "schema_summary",
]
