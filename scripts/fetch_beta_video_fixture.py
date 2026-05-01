#!/usr/bin/env python3
"""Stream-extract the Beta head_color video fixture used by v0.3 integration tests.

Builds ``data/agibot_beta_sample/675/882736/`` containing:

- ``proprio_stats.h5`` — sliced from the existing 936938 proprio (real Beta
  schema, real values, truncated to 879 frames to align with the upstream
  head_color frame count). The `proprio_stats/856286-902108.tar` blob this
  episode actually lives in is 48 GB on HF; sequential-scan extraction
  takes ~30 min for one file. The truncated-from-936938 fallback documents
  itself in the worklog and keeps fixture acquisition under a minute.
- ``videos/head_color.mp4`` — real upstream Beta video for episode 882736
  (av1, 640x480, 879 frames @ 30fps, ~8 MB), stream-extracted from
  ``observations/675/880749-912853.tar`` via HTTP Range requests against
  HF — only the 8 MB tar slice gets downloaded, not the full 36 GB tar.

Total fixture size: ~9 MB.

Requirements: ``HF_TOKEN`` with `agibot-world/AgiBotWorld-Beta` access
granted (gated dataset).

Usage::

    huggingface-cli login
    uv run python scripts/fetch_beta_video_fixture.py

Idempotent — re-running on an existing fixture exits without re-fetching.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import h5py
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO_ID = "agibot-world/AgiBotWorld-Beta"
TASK = "675"
EP_ID = "882736"
N_FRAMES = 879  # matches upstream head_color video for ep 882736

ROOT = Path("data/agibot_beta_sample")
DST_DIR = ROOT / TASK / EP_ID


def _ensure_video() -> None:
    video_dst = DST_DIR / "videos" / "head_color.mp4"
    if video_dst.is_file() and video_dst.stat().st_size > 1_000_000:
        print(f"OK: {video_dst} already exists ({video_dst.stat().st_size/1e6:.1f} MB)")
        return

    obs_tar = f"observations/{TASK}/880749-912853.tar"
    url = hf_hub_url(REPO_ID, obs_tar, repo_type="dataset")
    meta = get_hf_file_metadata(url)
    headers = build_hf_headers()
    headers["Range"] = "bytes=0-100000"
    req = urllib.request.Request(meta.location, headers=headers)
    with urllib.request.urlopen(req) as r:
        head = r.read()

    target = f"{EP_ID}/videos/head_color.mp4"
    offset = 0
    body_offset = None
    body_size = None
    while offset + 512 < len(head):
        name = head[offset : offset + 100].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if not name:
            offset += 512
            continue
        size_str = head[offset + 124 : offset + 136].split(b"\x00", 1)[0].decode().strip()
        size = int(size_str, 8) if size_str else 0
        if name == target:
            body_offset = offset + 512
            body_size = size
            break
        offset += 512 + ((size + 511) // 512) * 512

    if body_offset is None:
        raise RuntimeError(f"could not locate {target} in tar header window")

    headers = build_hf_headers()
    headers["Range"] = f"bytes={body_offset}-{body_offset + body_size - 1}"
    req = urllib.request.Request(meta.location, headers=headers)
    with urllib.request.urlopen(req) as r:
        body = r.read()

    video_dst.parent.mkdir(parents=True, exist_ok=True)
    video_dst.write_bytes(body)
    print(f"wrote {video_dst} ({len(body)/1e6:.2f} MB)")


def _ensure_proprio() -> None:
    proprio_dst = DST_DIR / "proprio_stats.h5"
    if proprio_dst.is_file():
        print(f"OK: {proprio_dst} already exists")
        return

    legacy = ROOT / TASK / "936938" / "proprio_stats.h5"
    if not legacy.is_file():
        raise RuntimeError(
            f"legacy proprio fixture {legacy} missing — re-acquire via Sprint 3 path"
        )

    proprio_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(legacy, proprio_dst)
    with h5py.File(proprio_dst, "r+") as f:
        paths: list[str] = []
        f.visit(paths.append)
        truncated = 0
        for p in paths:
            obj = f[p]
            if isinstance(obj, h5py.Dataset) and obj.shape and obj.shape[0] >= N_FRAMES:
                data = obj[:N_FRAMES]
                dtype = obj.dtype
                attrs = dict(obj.attrs)
                del f[p]
                ds = f.create_dataset(p, data=data, dtype=dtype)
                for k, v in attrs.items():
                    ds.attrs[k] = v
                truncated += 1
    size_mb = proprio_dst.stat().st_size / 1e6
    print(f"wrote {proprio_dst} ({size_mb:.2f} MB, {truncated} ds truncated)")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_video()
    _ensure_proprio()
    print(f"\nfixture ready at {DST_DIR}")


if __name__ == "__main__":
    main()
