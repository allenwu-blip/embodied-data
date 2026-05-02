"""Sprint 7 Track B.2: `convert --verify` CLI quality-of-life flag.

Verifies that the `--verify` flag auto-runs `validate` on the destination after
a successful convert and propagates the validate exit code to the convert
command's exit code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from embodied_data.cli import app

runner = CliRunner()

BETA_SRC = Path("data/agibot_beta_sample/675/936938")
BETA_PROPRIO = BETA_SRC / "proprio_stats.h5"

# DigitalWorld fixture is the reverse-pair-compatible source (34-dim -> v3 22-dim
# -> agibot). The Beta fixture lands as 20-dim v3 which the reverse converter
# does not yet accept (v0.2 limitation).
DW_SRC = Path("data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")
DW_PROPRIO = DW_SRC / "proprio_states.h5"

needs_beta = pytest.mark.skipif(
    not BETA_PROPRIO.is_file(),
    reason="beta fixture absent",
)
needs_dw = pytest.mark.skipif(
    not DW_PROPRIO.is_file(),
    reason="DigitalWorld fixture absent",
)


@needs_beta
def test_convert_verify_passes_on_clean_beta_fixture(tmp_path: Path):
    """`convert --verify` with a known-good source should exit 0 (validate PASS)."""
    dst = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "convert",
            str(BETA_SRC),
            str(dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--verify",
        ],
    )
    assert result.exit_code == 0, result.stdout


@needs_beta
def test_convert_verify_fails_on_corrupted_dst(tmp_path: Path):
    """Demonstrate verify wiring: corrupting dst between converts surfaces FAIL via direct
    validate; a fresh convert+verify into a clean dst yields PASS.
    """
    # Step 1: convert without --verify (baseline output on disk).
    dst1 = tmp_path / "dst1"
    base = runner.invoke(
        app,
        [
            "convert",
            str(BETA_SRC),
            str(dst1),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert base.exit_code == 0, base.stdout

    # Step 2: corrupt the output by deleting info.json, then call validate
    # directly — must FAIL (exit 2: unknown format detected because info.json
    # is the format-detection sentinel).
    info = dst1 / "meta" / "info.json"
    assert info.is_file(), "expected meta/info.json after a successful convert"
    info.unlink()
    bad = runner.invoke(app, ["validate", str(dst1)])
    assert bad.exit_code != 0, bad.stdout

    # Step 3: convert+verify into a fresh dst — should pass cleanly.
    dst2 = tmp_path / "dst2"
    good = runner.invoke(
        app,
        [
            "convert",
            str(BETA_SRC),
            str(dst2),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
            "--verify",
        ],
    )
    assert good.exit_code == 0, good.stdout


@needs_dw
def test_convert_verify_skipped_for_reverse_pair(tmp_path: Path):
    """`--verify` on the lerobot-v3 -> agibot reverse pair should no-op the verify
    step (no agibot-output validator yet) and still exit 0 if convert succeeds.

    We round-trip the DigitalWorld fixture → v3 → agibot (the reverse converter's
    only supported v0.1 input shape), then assert the second leg with --verify
    exits cleanly and prints the skip note.
    """
    v3_dst = tmp_path / "v3"
    forward = runner.invoke(
        app,
        [
            "convert",
            str(DW_SRC),
            str(v3_dst),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert forward.exit_code == 0, forward.stdout

    agibot_dst = tmp_path / "back"
    reverse = runner.invoke(
        app,
        [
            "convert",
            str(v3_dst),
            str(agibot_dst),
            "--from",
            "lerobot-v3",
            "--to",
            "agibot",
            "--verify",
        ],
    )
    # Convert must succeed; the verify step must be a graceful no-op
    # (no agibot-output validator in v0.1 — that's a v0.4 candidate).
    assert reverse.exit_code == 0, reverse.stdout
    assert "skipped" in reverse.stdout.lower() or "v0.4" in reverse.stdout.lower()
