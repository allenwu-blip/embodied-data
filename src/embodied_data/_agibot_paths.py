"""Filename-resolution helpers shared across convert / preview / validate.

Real AgiBot Beta captures use ``proprio_stats.h5`` (no second ``s`` in
"states") while the published DigitalWorld sim uses ``proprio_states.h5``.
v0.1.0 hard-coded the sim filename in ~7 call sites and silently failed on
real-Beta data. This module centralises the lookup so the two filenames are
treated as equivalent for *discovery purposes only* — schema-level guards
elsewhere still refuse Beta input cleanly (see ``convert.agibot_to_lerobot``).
"""

from __future__ import annotations

from pathlib import Path

# Glob pattern matching both filenames in a single ``Path.glob`` / ``rglob``
# call. Order in the character class is irrelevant; both letters are accepted.
PROPRIO_GLOB = "proprio_stat[se]*.h5"


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


__all__ = ["PROPRIO_GLOB", "find_proprio_h5", "iter_proprio_h5"]
