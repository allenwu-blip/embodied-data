"""Version-tag drift guard.

Reads `[project].version` from `pyproject.toml` and compares it against the
latest git tag matching `v*`. Always prints the comparison line. Exits 0 if:
  - no `v*` tag exists yet (pre-release state), or
  - the tag matches the pyproject version exactly.
Exits 1 if a `v*` tag exists and does not match.

Usage:
    python scripts/check_version.py
    python scripts/check_version.py --strict   # exit 1 even when no tag exists
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_project_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def latest_v_tag() -> str | None:
    """Return the latest tag matching `v*`, or None if no such tag exists.

    Picks the most recent tag by creation date (`--sort=-creatordate`) so that
    the result reflects the tag a release is actually about to push, even when
    earlier tags exist.
    """
    try:
        result = subprocess.run(
            ["git", "tag", "--list", "v*", "--sort=-creatordate"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    tags = [t for t in result.stdout.splitlines() if t.strip()]
    return tags[0] if tags else None


def normalize_tag(tag: str) -> str:
    """Strip leading `v` from a tag like `v0.1.0` -> `0.1.0`."""
    return re.sub(r"^v", "", tag)


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    version = read_project_version()
    tag = latest_v_tag()

    if tag is None:
        print(f"version={version} tag=<none> match=true")
        return 1 if strict else 0

    match = normalize_tag(tag) == version
    print(f"version={version} tag={tag} match={'true' if match else 'false'}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
