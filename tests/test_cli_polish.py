"""Tests for Sprint 2 CLI-polish features: --json, --version, inspect, error suggestions."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from embodied_data.cli import app

runner = CliRunner()

SAMPLE_META = Path(
    "data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96"
)
SAMPLE_H5 = SAMPLE_META / "proprio_states.h5"


# ---------------------------------------------------------------------------
# --version / version subcommand
# ---------------------------------------------------------------------------


def test_version_default_format():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "embodied-data" in result.stdout
    assert "sha=" in result.stdout
    assert "built=" in result.stdout


def test_version_json():
    result = runner.invoke(app, ["--json", "version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["version"]
    assert "git_sha" in payload
    assert "build_date" in payload


def test_top_level_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "embodied-data" in result.stdout


def test_top_level_version_flag_json():
    result = runner.invoke(app, ["--json", "--version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "version" in payload and "git_sha" in payload


# ---------------------------------------------------------------------------
# Global --json: validate / preview
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_H5.is_file(), reason="AgiBot fixture absent")
def test_global_json_validate_emits_json():
    result = runner.invoke(app, ["--json", "validate", str(SAMPLE_META)])
    # Allow exit 0 (PASS) or 1 (FAIL), but stdout must be JSON either way.
    assert result.exit_code in (0, 1)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["path"].endswith("000aa0b4-8fbe-432a-b6ae-559a7d7b3b96")
    assert payload["format"] == "agibot"
    assert isinstance(payload["results"], list)
    assert all({"name", "status", "detail"} <= set(r) for r in payload["results"])
    assert payload["result"] in ("PASS", "WARN", "FAIL")
    assert "exit_code" in payload


@pytest.mark.skipif(not SAMPLE_H5.is_file(), reason="AgiBot fixture absent")
def test_global_json_preview_emits_json():
    result = runner.invoke(app, ["--json", "preview", str(SAMPLE_META)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["format"] == "agibot"
    assert isinstance(payload["stats"], dict)
    assert "Total frames" in payload["stats"]


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_H5.is_file(), reason="AgiBot fixture absent")
def test_inspect_h5():
    result = runner.invoke(app, ["inspect", str(SAMPLE_H5)])
    assert result.exit_code == 0
    assert "state" in result.stdout
    assert "shape=" in result.stdout or "(group)" in result.stdout


@pytest.mark.skipif(not SAMPLE_H5.is_file(), reason="AgiBot fixture absent")
def test_inspect_h5_json():
    result = runner.invoke(app, ["--json", "inspect", str(SAMPLE_H5)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["kind"] == "h5"
    names = {n["name"] for n in payload["nodes"]}
    assert any("state" in n for n in names)


def test_inspect_parquet(tmp_path: Path):
    p = tmp_path / "tiny.parquet"
    table = pa.table({"a": pa.array([1, 2, 3]), "b": pa.array(["x", "y", "z"])})
    pq.write_table(table, p)
    result = runner.invoke(app, ["inspect", str(p)])
    assert result.exit_code == 0
    assert "rows: 3" in result.stdout
    assert "a" in result.stdout and "b" in result.stdout


def test_inspect_parquet_json(tmp_path: Path):
    p = tmp_path / "tiny.parquet"
    table = pa.table({"a": pa.array([1, 2, 3])})
    pq.write_table(table, p)
    result = runner.invoke(app, ["--json", "inspect", str(p)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["kind"] == "parquet"
    assert payload["num_rows"] == 3
    assert payload["schema"][0]["name"] == "a"


def test_inspect_unknown_extension(tmp_path: Path):
    p = tmp_path / "thing.txt"
    p.write_text("hi")
    result = runner.invoke(app, ["inspect", str(p)])
    assert result.exit_code == 2
    assert "unsupported" in result.stdout.lower() or "suggestion" in result.stdout.lower()


def test_inspect_missing_path(tmp_path: Path):
    result = runner.invoke(app, ["inspect", str(tmp_path / "nope.h5")])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Error suggestions
# ---------------------------------------------------------------------------


def test_validate_error_suggestion(tmp_path: Path):
    bogus = tmp_path / "does_not_exist"
    result = runner.invoke(app, ["validate", str(bogus)])
    assert result.exit_code == 2
    out = result.stdout.lower()
    assert "suggestion" in out
    assert "exist" in out


def test_validate_unknown_format(tmp_path: Path):
    # An empty existing dir → format detection fails.
    tmp_path.mkdir(exist_ok=True)
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 2
    assert "suggestion" in result.stdout.lower()
    assert "--format" in result.stdout


def test_preview_path_missing(tmp_path: Path):
    result = runner.invoke(app, ["preview", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "suggestion" in result.stdout.lower()


def test_validate_error_json(tmp_path: Path):
    bogus = tmp_path / "missing"
    result = runner.invoke(app, ["--json", "validate", str(bogus)])
    assert result.exit_code == 2
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "error" in payload
    assert payload["exit_code"] == 2
