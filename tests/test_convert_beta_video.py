"""v0.3 integration tests for AgiBot Beta head_color video pipeline.

Covers the public contract of v0.3.0:

- ``probe_video`` returns sensible metadata on real Beta upstream mp4
- ``reencode_video`` lands an h264 file with bf=0/g=2/yuv420p/monotonic PTS
- single-episode + batch convert emit
  ``observation.images.head_color`` features + per-episode video columns
- legacy proprio-only fixtures continue to work (no video declared)
- validate FAILs when info.features declares video but the file is
  missing or broken (no longer SKIPs)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embodied_data.convert._video import (
    count_reencoded_frames,
    probe_video,
    reencode_video,
)
from embodied_data.convert.agibot_beta_to_lerobot import (
    HEAD_CAMERA_KEY,
    convert_agibot_beta_batch,
    convert_agibot_beta_to_lerobot_v3,
    find_head_color_video,
)
from embodied_data.validate import lerobot_v3 as v3_validate

BETA_VIDEO_FIXTURE = Path("data/agibot_beta_sample/675/882736/videos/head_color.mp4")
BETA_VIDEO_EPISODE = Path("data/agibot_beta_sample/675/882736")
BETA_LEGACY_EPISODE = Path("data/agibot_beta_sample/675/936938")
BETA_TASK_INFO = Path("data/agibot_beta_sample/task_info_675.json")

needs_video_fixture = pytest.mark.skipif(
    not BETA_VIDEO_FIXTURE.is_file(),
    reason="Beta video fixture absent (run scripts/fetch_beta_video_fixture.py)",
)
needs_legacy_fixture = pytest.mark.skipif(
    not BETA_LEGACY_EPISODE.is_dir(),
    reason="Beta legacy fixture absent",
)


# ---------------------------------------------------------------------------
# probe_video / reencode_video unit tests
# ---------------------------------------------------------------------------


@needs_video_fixture
def test_probe_video_returns_sensible_metadata():
    meta = probe_video(BETA_VIDEO_FIXTURE)
    # Beta upstream is av1 @ 30fps, 640x480; we just assert sensible bounds so
    # the test stays robust if upstream re-encodes their fixture.
    assert meta.width >= 320 and meta.width <= 4096
    assert meta.height >= 240 and meta.height <= 4096
    assert meta.fps == pytest.approx(30.0, abs=0.5)
    # Codec name comes from the libav decoder identifier, e.g. ``libdav1d``
    # for av1 or ``h264`` for h264. Just assert it's a non-empty string.
    assert isinstance(meta.codec_name, str) and meta.codec_name


@needs_video_fixture
def test_reencode_video_lands_h264_with_lerobot_v3_constraints(tmp_path: Path):
    out = tmp_path / "head_color.mp4"
    reencode_video(BETA_VIDEO_FIXTURE, out, fps=30)
    assert out.is_file()
    with av.open(str(out)) as c:
        s = c.streams.video[0]
        ctx = s.codec_context
        assert ctx.codec.name == "h264"
        assert ctx.pix_fmt == "yuv420p"
        # No B-frames means DTS == PTS; spot-check first ~30 packets.
        c.seek(0)
        seen = 0
        prev_pts = -1
        for packet in c.demux(s):
            if packet.pts is None:
                continue
            assert packet.pts > prev_pts, "pts must be monotonic"
            prev_pts = packet.pts
            seen += 1
            if seen >= 30:
                break
        assert seen > 0


@needs_video_fixture
def test_count_reencoded_frames_matches_upstream(tmp_path: Path):
    src_meta = probe_video(BETA_VIDEO_FIXTURE)
    out = tmp_path / "head_color.mp4"
    reencode_video(BETA_VIDEO_FIXTURE, out, fps=30)
    n = count_reencoded_frames(out)
    # Upstream container may report 0 frames for av1; if so, fall back to
    # duration-based estimation.
    expected = src_meta.n_frames if src_meta.n_frames > 0 else round(src_meta.duration * 30)
    assert abs(n - expected) <= 1


# ---------------------------------------------------------------------------
# Single-episode integration
# ---------------------------------------------------------------------------


@needs_video_fixture
def test_single_episode_with_video_emits_head_color_feature(tmp_path: Path):
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)

    # Video file landed at the v3 canonical location.
    video_dst = dst / "videos" / HEAD_CAMERA_KEY / "chunk-000" / "file-000.mp4"
    assert video_dst.is_file()

    # info.json declares the head_color video feature + populates video_path.
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert HEAD_CAMERA_KEY in info["features"]
    assert info["features"][HEAD_CAMERA_KEY]["dtype"] == "video"
    assert info["video_path"] is not None
    assert "{video_key}" in info["video_path"]

    # Episode meta carries the video columns.
    ep_meta = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    cols = set(ep_meta.schema.names)
    for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
        assert f"videos/{HEAD_CAMERA_KEY}/{suffix}" in cols


@needs_video_fixture
def test_single_episode_with_video_validates_pass(tmp_path: Path):
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)
    info = json.loads((dst / "meta" / "info.json").read_text())
    result = v3_validate.check_alignment(dst, info)
    assert result.status == "PASS", result.detail


@needs_legacy_fixture
def test_single_episode_no_video_legacy_path(tmp_path: Path):
    """Proprio-only fixtures still produce v0.2-shape output."""
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_LEGACY_EPISODE, dst=dst)
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert HEAD_CAMERA_KEY not in info["features"]
    assert info["video_path"] is None
    assert not (dst / "videos").exists()


def test_find_head_color_video_returns_none_when_missing(tmp_path: Path):
    assert find_head_color_video(tmp_path) is None


def test_find_head_color_video_returns_path_when_present(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    target = tmp_path / "videos" / "head_color.mp4"
    target.write_bytes(b"")  # presence check only
    assert find_head_color_video(tmp_path) == target


# ---------------------------------------------------------------------------
# Batch integration
# ---------------------------------------------------------------------------


@needs_video_fixture
@needs_legacy_fixture
def test_batch_mixed_with_and_without_video_logs_failure(tmp_path: Path):
    """Batch with one video-bearing episode + one legacy episode emits the
    video dataset, logs the legacy episode to .beta_batch_errors.jsonl,
    and produces no orphan parquets."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    shutil.copy(BETA_TASK_INFO, src_root / BETA_TASK_INFO.name)
    (src_root / "675").mkdir()
    shutil.copytree(BETA_VIDEO_EPISODE, src_root / "675" / BETA_VIDEO_EPISODE.name)
    shutil.copytree(BETA_LEGACY_EPISODE, src_root / "675" / BETA_LEGACY_EPISODE.name)

    dst = tmp_path / "v3"
    convert_agibot_beta_batch(src=src_root, dst=dst)

    error_log = dst / ".beta_batch_errors.jsonl"
    assert error_log.is_file()
    errors = [json.loads(line) for line in error_log.read_text().splitlines()]
    assert len(errors) == 1
    assert errors[0]["episode_id"] == BETA_LEGACY_EPISODE.name
    assert "head_color" in errors[0]["error"]

    # No orphan parquet for the failed episode.
    data_files = sorted((dst / "data" / "chunk-000").glob("file-*.parquet"))
    ep_meta_files = sorted((dst / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    video_files = sorted((dst / "videos" / HEAD_CAMERA_KEY / "chunk-000").glob("file-*.mp4"))
    assert len(data_files) == len(ep_meta_files) == len(video_files) == 1


@needs_legacy_fixture
def test_batch_no_videos_anywhere_keeps_v02_behavior(tmp_path: Path):
    """When no episode has video, batch produces a proprio-only dataset."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    shutil.copy(BETA_TASK_INFO, src_root / BETA_TASK_INFO.name)
    (src_root / "675").mkdir()
    shutil.copytree(BETA_LEGACY_EPISODE, src_root / "675" / BETA_LEGACY_EPISODE.name)

    dst = tmp_path / "v3"
    convert_agibot_beta_batch(src=src_root, dst=dst)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert HEAD_CAMERA_KEY not in info["features"]
    assert info["video_path"] is None
    assert not (dst / "videos").exists()


# ---------------------------------------------------------------------------
# validate negative-path tests
# ---------------------------------------------------------------------------


@needs_video_fixture
def test_validate_fails_when_declared_video_file_missing(tmp_path: Path):
    """Convert with video, then delete the videos dir → validate FAIL."""
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)
    shutil.rmtree(dst / "videos")
    info = json.loads((dst / "meta" / "info.json").read_text())
    result = v3_validate.check_alignment(dst, info)
    assert result.status == "FAIL"
    assert "missing" in result.detail.lower() or "head_color" in result.detail


@needs_video_fixture
def test_validate_fails_when_declared_but_no_episode_video_columns(tmp_path: Path):
    """info declares video but episode meta lacks video columns → FAIL."""
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)

    # Strip video columns from episode meta but keep info.features intact.
    ep_meta_path = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(ep_meta_path)
    keep = [n for n in table.schema.names if not n.startswith("videos/")]
    pq.write_table(table.select(keep), ep_meta_path)

    info = json.loads((dst / "meta" / "info.json").read_text())
    result = v3_validate.check_alignment(dst, info)
    assert result.status == "FAIL"
    assert "from_timestamp" in result.detail or "video features" in result.detail


@needs_video_fixture
def test_validate_skips_when_no_video_declared(tmp_path: Path):
    """Proprio-only output keeps SKIP semantics — no false positive FAIL."""
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_LEGACY_EPISODE, dst=dst)
    info = json.loads((dst / "meta" / "info.json").read_text())
    result = v3_validate.check_alignment(dst, info)
    assert result.status == "SKIP"


@needs_video_fixture
def test_video_frame_count_mismatch_raises(tmp_path: Path):
    """If proprio frame count diverges from video by >1 frame, hard-fail."""
    # Build a fixture with mismatched proprio: copy 936938 (1090 frames) into
    # the same dir as the 879-frame head_color.mp4.
    fixture = tmp_path / "ep"
    (fixture / "videos").mkdir(parents=True)
    shutil.copy(BETA_VIDEO_FIXTURE, fixture / "videos" / "head_color.mp4")
    shutil.copy(BETA_LEGACY_EPISODE / "proprio_stats.h5", fixture / "proprio_stats.h5")
    # Need a task_info.json beside it for task name resolution; reuse upstream's.
    shutil.copy(BETA_TASK_INFO, fixture.parent / BETA_TASK_INFO.name)

    with pytest.raises(ValueError, match="frame count"):
        convert_agibot_beta_to_lerobot_v3(src=fixture, dst=tmp_path / "v3")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


@needs_video_fixture
def test_cli_convert_with_video_then_validate_pass(tmp_path: Path):
    """End-to-end CLI: convert single-episode with video, then validate exits 0."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    runner = CliRunner()
    dst = tmp_path / "v3"
    r1 = runner.invoke(
        app,
        [
            "convert",
            str(BETA_VIDEO_EPISODE),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["validate", str(dst)])
    assert r2.exit_code == 0, r2.output


# Smoke check that ffprobe agrees on the v3 video (catches subtle muxer bugs).
@needs_video_fixture
def test_ffprobe_confirms_no_b_frames(tmp_path: Path):
    """Independent codec assertion via ffprobe rather than PyAV."""
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not installed")
    out = tmp_path / "head_color.mp4"
    reencode_video(BETA_VIDEO_FIXTURE, out, fps=30)
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,has_b_frames,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "codec_name=h264" in res.stdout
    assert "has_b_frames=0" in res.stdout
    assert "pix_fmt=yuv420p" in res.stdout


@needs_video_fixture
def test_validate_passes_on_multi_episode_per_video_pattern(tmp_path: Path):
    """Regression: validate must PASS when multiple episodes share a single
    video file and slice via from_timestamp / to_timestamp.

    LeRobot's pusht packs all 206 episodes into one mp4 (25650 frames) with
    per-episode (from_ts, to_ts) ranges in episode meta. v0.3.0's frame-count
    cross-check incorrectly compared whole-video frames to single-episode
    length, producing 206 false-FAILs on the official quick-start dataset.
    v0.3.1 scopes the frame-count check to one-episode-per-mp4 cases only;
    multi-episode-per-mp4 relies on the duration check instead.
    """
    import shutil

    # Build a synthetic v3 dataset that mirrors the multi-episode-per-mp4
    # pattern by manually editing 882736's episode meta to claim two episodes
    # share the existing mp4 (one covers frames 0..439 -> 0..14.633s, the
    # other 440..878 -> 14.667..29.267s).
    dst = tmp_path / "v3"
    convert_agibot_beta_to_lerobot_v3(src=BETA_VIDEO_EPISODE, dst=dst)

    info = json.loads((dst / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    ep_meta_path = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(ep_meta_path)
    cols = {n: table.column(n).to_pylist() for n in table.schema.names}

    # Split the single 879-frame episode into two halves both pointing at the
    # same mp4 file. Left episode: rows 0..439. Right episode: rows 440..878.
    left_n, right_n = 440, 439
    cols["episode_index"] = [0, 1]
    cols["tasks"] = [cols["tasks"][0], cols["tasks"][0]]
    cols["length"] = [left_n, right_n]
    cols["meta/episodes/chunk_index"] = [0, 0]
    cols["meta/episodes/file_index"] = [0, 0]
    cols["data/chunk_index"] = [0, 0]
    cols["data/file_index"] = [0, 0]
    cols["dataset_from_index"] = [0, left_n]
    cols["dataset_to_index"] = [left_n, left_n + right_n]
    # Point both at the same mp4 file (file_index=0); slice via timestamps.
    cols[f"videos/{HEAD_CAMERA_KEY}/chunk_index"] = [0, 0]
    cols[f"videos/{HEAD_CAMERA_KEY}/file_index"] = [0, 0]
    cols[f"videos/{HEAD_CAMERA_KEY}/from_timestamp"] = [0.0, left_n / fps]
    cols[f"videos/{HEAD_CAMERA_KEY}/to_timestamp"] = [left_n / fps, (left_n + right_n) / fps]

    # Stats columns: clone the original row's stats for both episodes (they're
    # not what the alignment check looks at — duration / frame count is).
    for k, v in list(cols.items()):
        if k.startswith("stats/"):
            cols[k] = [v[0], v[0]]

    new_table = pa.Table.from_pydict(cols)
    pq.write_table(new_table, ep_meta_path)

    # Frame-video alignment must PASS — duration check sees per-episode slices,
    # frame-count check correctly skips multi-episode-per-mp4 case.
    result = v3_validate.check_alignment(dst, info)
    assert result.status == "PASS", result.detail

    # Sanity: shutil unused → reference it so ruff doesn't flag.
    _ = shutil
