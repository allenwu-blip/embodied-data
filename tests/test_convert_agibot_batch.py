"""Integration tests for the AgiBot → LeRobot v3 batch converter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import av
import pyarrow.parquet as pq

from embodied_data.convert.agibot_to_lerobot import convert_agibot_batch


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _video_signature(p: Path) -> tuple[int, float]:
    """ffprobe-equivalent: (frame_count, duration_seconds) via PyAV."""
    container = av.open(str(p))
    try:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    finally:
        container.close()
    return frames, duration


def _all_episode_rows(dst: Path) -> list[dict]:
    rows: list[dict] = []
    for parquet in sorted((dst / "meta" / "episodes").rglob("file-*.parquet")):
        t = pq.read_table(parquet)
        for i in range(t.num_rows):
            rows.append({col: t[col][i].as_py() for col in t.column_names})
    return rows


def test_batch_10_episodes(synthetic_agibot_root, tmp_path):
    out = tmp_path / "out"
    convert_agibot_batch(src=synthetic_agibot_root, dst=out)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 10
    assert info["total_frames"] == 3750  # 10 × 375
    assert info["total_tasks"] == 1

    # All episode parquets land in chunk-000 (well under 100MB).
    chunks = sorted((out / "data").iterdir())
    assert [c.name for c in chunks] == ["chunk-000"]

    rows = _all_episode_rows(out)
    assert len(rows) == 10
    assert {r["episode_index"] for r in rows} == set(range(10))
    assert all(r["length"] == 375 for r in rows)
    # Global frame-index ranges should be contiguous and non-overlapping.
    rows.sort(key=lambda r: r["episode_index"])
    expected_from = 0
    for r in rows:
        assert r["dataset_from_index"] == expected_from
        assert r["dataset_to_index"] == expected_from + 375
        expected_from += 375

    # uuid map persisted under meta/extra/.
    uuid_map = pq.read_table(out / "meta" / "extra" / "uuid_map.parquet")
    assert uuid_map.num_rows == 10
    assert set(uuid_map["agibot_task"].to_pylist()) == {"digitaltwin_3"}
    assert len(set(uuid_map["agibot_uuid"].to_pylist())) == 10


def test_batch_max_episodes(synthetic_agibot_root, tmp_path):
    out = tmp_path / "out"
    convert_agibot_batch(src=synthetic_agibot_root, dst=out, max_episodes=3)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 1125

    rows = _all_episode_rows(out)
    assert len(rows) == 3


def test_batch_resume_idempotent(synthetic_agibot_root, tmp_path):
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    # Clean run A.
    convert_agibot_batch(src=synthetic_agibot_root, dst=out_a, max_episodes=4)

    # Half-baked run B → resume to completion.
    convert_agibot_batch(src=synthetic_agibot_root, dst=out_b, max_episodes=2)
    convert_agibot_batch(src=synthetic_agibot_root, dst=out_b, max_episodes=4, resume=True)

    # Parquet files must be byte-identical (deterministic writes).
    parquet_paths = sorted(p.relative_to(out_a) for p in out_a.rglob("*.parquet"))
    assert parquet_paths == sorted(p.relative_to(out_b) for p in out_b.rglob("*.parquet"))
    for rel in parquet_paths:
        assert _hash_file(out_a / rel) == _hash_file(out_b / rel), f"diverged at {rel}"

    # info.json must match exactly.
    assert (out_a / "meta" / "info.json").read_text() == (out_b / "meta" / "info.json").read_text()
    # stats.json: aggregate stats are deterministic from the same source.
    assert (out_a / "meta" / "stats.json").read_text() == (
        out_b / "meta" / "stats.json"
    ).read_text()

    # Videos: compare by ffprobe signature (mtime/encoder metadata may differ).
    video_paths = sorted(p.relative_to(out_a) for p in out_a.rglob("*.mp4"))
    assert video_paths == sorted(p.relative_to(out_b) for p in out_b.rglob("*.mp4"))
    for rel in video_paths:
        assert _video_signature(out_a / rel) == _video_signature(out_b / rel), (
            f"video signature diverged at {rel}"
        )


def test_batch_progress_emits(synthetic_agibot_root, tmp_path):
    """Light check that the rich progress bar fires per episode."""
    out = tmp_path / "out"
    real_update = __import__("rich.progress", fromlist=["Progress"]).Progress.update
    with mock.patch("rich.progress.Progress.update", autospec=True, side_effect=real_update) as m:
        convert_agibot_batch(src=synthetic_agibot_root, dst=out, max_episodes=2)
    # Updates land twice per episode (stage label + advance) → at least 2.
    assert m.call_count >= 2
