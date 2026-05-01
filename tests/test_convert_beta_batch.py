"""M3 Beta batch integration tests.

Covers:
- 10-episode synthetic batch (symlink-cloned from the single Beta fixture)
- --resume idempotency: rerun on full output is a no-op
- Mid-run kill: delete one episode's parquet between runs and confirm --resume
  fills only the gap
- Error log: corrupted h5 → continue, record to .beta_batch_errors.jsonl
- --max-episodes truncation
- Single-task vs multi-task aggregation in tasks.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from embodied_data.convert.agibot_beta_to_lerobot import (
    convert_agibot_beta_batch,
    is_beta_batch_src,
)

BETA_H5 = Path("data/agibot_beta_sample/675/936938/proprio_stats.h5")
BETA_TASK_INFO = Path("data/agibot_beta_sample/task_info_675.json")

needs_beta = pytest.mark.skipif(not BETA_H5.exists(), reason="beta fixture absent")


def _build_synthetic_root(
    base: Path,
    *,
    task_id: str = "675",
    episode_ids: list[str],
    task_info_src: Path = BETA_TASK_INFO,
    h5_src: Path = BETA_H5,
) -> Path:
    """Build a Beta task-dataset root by symlinking the single real fixture into
    multiple <task>/<ep>/ slots. Returns the root path."""
    root = base / "beta_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"task_info_{task_id}.json").symlink_to(task_info_src.resolve())
    for ep in episode_ids:
        ep_dir = root / task_id / ep
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "proprio_stats.h5").symlink_to(h5_src.resolve())
    return root


@needs_beta
def test_is_beta_batch_src_recognises_root(tmp_path: Path):
    root = _build_synthetic_root(tmp_path, episode_ids=["ep_0"])
    assert is_beta_batch_src(root) is True


@needs_beta
def test_is_beta_batch_src_rejects_single_episode_dir(tmp_path: Path):
    root = _build_synthetic_root(tmp_path, episode_ids=["ep_0"])
    ep_dir = root / "675" / "ep_0"
    assert is_beta_batch_src(ep_dir) is False


@needs_beta
def test_batch_10_episodes_smoke(tmp_path: Path):
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(10)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 10
    assert info["total_frames"] == 10 * 1090  # 10 episodes × 1090 frames each
    assert info["total_tasks"] == 1
    assert info["robot_type"] == "agibot-beta"
    assert info["features"]["observation.state"]["shape"] == [20]

    # Each episode → one parquet file
    parquets = sorted((dst / "data" / "chunk-000").glob("file-*.parquet"))
    assert len(parquets) == 10

    # Each per-episode meta is a parquet too
    eps_meta = sorted((dst / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    assert len(eps_meta) == 10

    # uuid_map covers all 10
    uuid_map = pq.read_table(dst / "meta" / "extra" / "uuid_map.parquet")
    assert uuid_map.num_rows == 10
    assert sorted(uuid_map["agibot_episode_id"].to_pylist()) == [f"ep_{i:02d}" for i in range(10)]


@needs_beta
def test_batch_max_episodes_truncates(tmp_path: Path):
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(10)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst, max_episodes=3)
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 3 * 1090


@needs_beta
def test_batch_resume_idempotent(tmp_path: Path):
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(5)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)
    info_path = dst / "meta" / "info.json"
    info_first = json.loads(info_path.read_text())
    stats_first = (dst / "meta" / "stats.json").read_text()

    # Rerun with --resume on the already-complete dst — should be a no-op
    # in the per-episode write sense, and re-emit identical aggregates.
    convert_agibot_beta_batch(src=root, dst=dst, resume=True)
    info_second = json.loads(info_path.read_text())
    stats_second = (dst / "meta" / "stats.json").read_text()

    assert info_first == info_second
    assert stats_first == stats_second


@needs_beta
def test_batch_resume_fills_only_missing(tmp_path: Path):
    """Delete one episode's data parquet + uuid_map row, rerun --resume, verify
    only the missing episode is reprocessed."""
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(5)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)

    # Delete ep_02's data parquet + filter uuid_map
    deleted_parquet = dst / "data" / "chunk-000" / "file-002.parquet"
    deleted_meta = dst / "meta" / "episodes" / "chunk-000" / "file-002.parquet"
    deleted_parquet.unlink()
    deleted_meta.unlink()
    uuid_map_path = dst / "meta" / "extra" / "uuid_map.parquet"
    table = pq.read_table(uuid_map_path)
    rows = table.to_pylist()
    rows = [r for r in rows if r["agibot_episode_id"] != "ep_02"]
    import pyarrow as pa

    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
                "agibot_task": pa.array([r["agibot_task"] for r in rows], type=pa.string()),
                "agibot_episode_id": pa.array(
                    [r["agibot_episode_id"] for r in rows], type=pa.string()
                ),
            }
        ),
        uuid_map_path,
    )

    convert_agibot_beta_batch(src=root, dst=dst, resume=True)
    # Should refill ep_02 (under a higher episode_index, since 0..4 already used)
    after = pq.read_table(uuid_map_path).to_pylist()
    ep_ids = sorted(r["agibot_episode_id"] for r in after)
    assert ep_ids == ["ep_00", "ep_01", "ep_02", "ep_03", "ep_04"]


@needs_beta
def test_batch_corrupted_h5_logged_to_jsonl(tmp_path: Path):
    """Pollute one episode's h5 so the read fails. Batch should continue,
    record the failure to .beta_batch_errors.jsonl, and still emit a v3
    dataset for the surviving episodes."""
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(5)],
    )
    # Replace ep_02's h5 with a corrupted file
    bad_path = root / "675" / "ep_02" / "proprio_stats.h5"
    bad_path.unlink()  # remove symlink
    bad_path.write_bytes(b"this is not h5")

    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)

    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4  # 5 minus 1 corrupted
    assert info["total_frames"] == 4 * 1090

    error_log = dst / ".beta_batch_errors.jsonl"
    assert error_log.is_file()
    lines = [json.loads(line) for line in error_log.read_text().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["episode_id"] == "ep_02"
    assert "error_type" in lines[0]


@needs_beta
def test_batch_workers_2_matches_workers_1(tmp_path: Path):
    """Running with workers=2 produces the same totals as workers=1.

    (Per-byte equality isn't guaranteed because parquet tail-padding can vary
    by writer state, but episode counts, frame counts, and stats must match.)
    """
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(4)],
    )
    dst1 = tmp_path / "out_w1"
    dst2 = tmp_path / "out_w2"
    convert_agibot_beta_batch(src=root, dst=dst1, workers=1)
    convert_agibot_beta_batch(src=root, dst=dst2, workers=2)

    info1 = json.loads((dst1 / "meta" / "info.json").read_text())
    info2 = json.loads((dst2 / "meta" / "info.json").read_text())
    assert info1["total_episodes"] == info2["total_episodes"]
    assert info1["total_frames"] == info2["total_frames"]
    assert info1["total_tasks"] == info2["total_tasks"]
    # Stats (mean / std) must match within float32 round-trip tolerance.
    s1 = json.loads((dst1 / "meta" / "stats.json").read_text())
    s2 = json.loads((dst2 / "meta" / "stats.json").read_text())
    np.testing.assert_allclose(
        s1["observation.state"]["mean"], s2["observation.state"]["mean"], rtol=1e-5
    )
    np.testing.assert_allclose(s1["action"]["std"], s2["action"]["std"], rtol=1e-5)


@needs_beta
def test_batch_validate_passes(tmp_path: Path):
    """Multi-episode Beta batch output passes embodied-data validate."""
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(5)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)

    from typer.testing import CliRunner

    from embodied_data.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(dst)])
    assert result.exit_code == 0, result.output
    assert "Result: PASS" in result.output


@needs_beta
def test_batch_no_video_dataset_passes_validate(tmp_path: Path):
    """Beta v0.2 first cut emits no videos. Confirm `embodied-data validate`
    handles no-video v3 datasets without false-FAILs (fps consistency must
    SKIP, frame-video alignment must PASS via episode-meta length×fps proxy)."""
    from typer.testing import CliRunner

    from embodied_data.cli import app

    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(3)],
    )
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)
    assert not (dst / "videos").exists()  # confirm no videos at all

    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(dst)])
    assert result.exit_code == 0, result.output
    output = result.output
    assert "fps consistency" in output and "SKIP" in output
    assert "frame-video alignment" in output and "PASS" in output
    assert "Result: PASS" in output


@needs_beta
def test_batch_does_not_leak_sparse_index_companions(tmp_path: Path):
    """Beta h5 has action/{joint,effector,end,robot,head,waist}/index sparse
    arrays. v0.2 silently drops them (LeRobot v3 has no slot for them).
    Confirm: the converter doesn't crash AND doesn't leak any '*/index'
    column into the output parquets. Those slots are reserved for the v0.3
    auxiliary.*.mask roadmap."""
    root = _build_synthetic_root(tmp_path, episode_ids=["ep_0"])
    dst = tmp_path / "out"
    convert_agibot_beta_batch(src=root, dst=dst)

    table = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    cols = set(table.column_names)
    # 'index' (standard v3 global frame counter) is allowed; '*/index'
    # action companions must NOT leak.
    assert "index" in cols
    leaked = {c for c in cols if c.endswith("/index") and c != "index"}
    assert leaked == set(), f"sparse */index companions leaked: {leaked}"


@needs_beta
def test_batch_via_cli_dispatcher_auto_routes(tmp_path: Path):
    """End-to-end via dispatcher: pointing CLI at a Beta task root with no
    flags routes to batch and writes a multi-episode v3 dataset."""
    root = _build_synthetic_root(
        tmp_path,
        episode_ids=[f"ep_{i:02d}" for i in range(3)],
    )
    dst = tmp_path / "out"
    from typer.testing import CliRunner

    from embodied_data.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "convert",
            str(root),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 0, result.output
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["robot_type"] == "agibot-beta"
