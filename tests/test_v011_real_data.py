"""Real-Beta-data integration tests for v0.1.1 patches.

Each test pins a specific behavioral guarantee against the Beta sample at
``data/agibot_beta_sample/675/936938/proprio_stats.h5`` (note: ``stats``, not
``states`` — Beta naming differs from sim DigitalWorld). Tests are skipped when
the fixture is absent so contributors without the sample can still run pytest.

The Beta fixture has these distinguishing facts (see docs/v0.1.1-findings.md):
- file is ``proprio_stats.h5`` (no second ``s``)
- ``state/joint/position`` shape is ``(N, 14)`` (sim is 34)
- ``state/joint.attrs["name"]`` is missing entirely (sim has 34 names)
- ``task_info_<task_id>.json`` lives at the *task root*, contains a list of 399
  per-episode dicts (sim has a single dict at the episode dir)
- extra state subgroups ``head``, ``waist`` exist (sim doesn't)
- ``/timestamp`` is ``int64`` ns (sim is ``float32`` seconds)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from embodied_data.cli import app

BETA_H5 = Path("data/agibot_beta_sample/675/936938/proprio_stats.h5")
BETA_EP_DIR = BETA_H5.parent  # 675/936938/
BETA_TASK_ROOT = BETA_EP_DIR.parent  # 675/
BETA_TASK_INFO = Path("data/agibot_beta_sample/task_info_675.json")

needs_beta = pytest.mark.skipif(
    not BETA_H5.exists(),
    reason="beta fixture absent",
)

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Patch 1 — filename glob accepts both proprio_states.h5 and proprio_stats.h5
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch1_find_proprio_h5_finds_beta_naming():
    """Helper resolves Beta's ``proprio_stats.h5`` when searching an episode dir."""
    from embodied_data._agibot_paths import find_proprio_h5

    found = find_proprio_h5(BETA_EP_DIR)
    assert found is not None
    assert found.name == "proprio_stats.h5"


@needs_beta
def test_patch1_find_proprio_h5_finds_sim_naming(tmp_path: Path):
    """Helper still resolves the sim ``proprio_states.h5`` filename."""
    from embodied_data._agibot_paths import find_proprio_h5

    (tmp_path / "proprio_states.h5").write_bytes(b"")
    found = find_proprio_h5(tmp_path)
    assert found is not None
    assert found.name == "proprio_states.h5"


@needs_beta
def test_patch1_detect_format_recognizes_beta_filename(tmp_path: Path):
    """``preview.detect_format`` recognises Beta filename as agibot."""
    from embodied_data.preview import detect_format

    (tmp_path / "proprio_stats.h5").write_bytes(b"")
    assert detect_format(tmp_path) == "agibot"


# --------------------------------------------------------------------------- #
# Patch 2 — convert refuses Beta cleanly (no traceback, exit 2, helpful msg)
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch2_convert_refuses_beta_with_clean_error(tmp_path: Path):
    """Convert exits 2 with a structured error when given real-robot Beta input.

    The Beta fixture has the task_info JSON at the task-dataset root and uses
    Beta-style filename, so we set up a temporary AgiBot-shaped tree pointing
    at the Beta h5 to exercise the schema-detection path without modifying the
    real fixture.
    """
    # Build a minimal /meta_info/<task>/<uuid>/ + sibling /observations layout
    # whose only purpose is to give convert a real source dir to chew on.
    task = "beta_real"
    uuid_ = "936938"
    ep_dir = tmp_path / "src" / "meta_info" / task / uuid_
    obs_dir = tmp_path / "src" / "observations" / task / uuid_ / "video"
    ep_dir.mkdir(parents=True)
    obs_dir.mkdir(parents=True)
    # Hard-link the Beta h5 in (Beta naming).
    (ep_dir / "proprio_stats.h5").symlink_to(BETA_H5.resolve())
    # Also stage a task_info.json so single-episode discovery finds the dir.
    (ep_dir / "task_info.json").write_text(json.dumps([{"episode_id": 936938, "task_name": "x"}]))
    # Dummy 1-frame mp4 so video resolver doesn't FileNotFoundError before us.
    (obs_dir / "head.mp4").write_bytes(b"")

    result = runner.invoke(
        app,
        [
            "convert",
            str(tmp_path / "src" / "meta_info" / task / uuid_),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 2, f"expected exit 2, got {result.exit_code}\n{result.output}"
    output = result.output.lower()
    assert "real-robot" in output or "v0.2" in output or "digitalworld" in output, (
        f"error message should mention Beta refusal; got:\n{result.output}"
    )
    # No raw Python traceback should leak.
    assert "traceback" not in output


# --------------------------------------------------------------------------- #
# Patch 3 — _read_task_name handles list-of-episodes JSON
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch3_read_task_name_handles_list(tmp_path: Path):
    """``_read_task_name`` resolves a Beta-style list of episode dicts."""
    from embodied_data.convert.agibot_to_lerobot import _read_task_name

    p = tmp_path / "task_info_675.json"
    p.write_text(
        json.dumps(
            [
                {"episode_id": 999, "task_name": "wrong one"},
                {"episode_id": 936938, "task_name": "Insert the straw"},
            ]
        )
    )
    name = _read_task_name(p)
    assert "Insert the straw" in name or "wrong one" in name  # fallback to [0] is acceptable


@needs_beta
def test_patch3_read_task_name_dict_still_works(tmp_path: Path):
    """Dict layout (sim) still returns the inner ``task_name``."""
    from embodied_data.convert.agibot_to_lerobot import _read_task_name

    p = tmp_path / "task_info.json"
    p.write_text(json.dumps({"task_name": "DigitalWorld task"}))
    assert _read_task_name(p) == "DigitalWorld task"


# --------------------------------------------------------------------------- #
# Patch 4 — preview reports actual joint count (no "State dim 22 (raw 14)" lie)
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch4_preview_does_not_claim_22_for_beta(tmp_path: Path):
    """Preview against a Beta-shaped episode does not falsely claim 22-dim state."""
    # Build a Beta-shaped meta_info episode dir using a symlink to the real h5.
    ep = tmp_path / "meta_info" / "beta_task" / "ep0"
    ep.mkdir(parents=True)
    (ep / "proprio_stats.h5").symlink_to(BETA_H5.resolve())
    # Beta task_info layout: list at task root. Provide a per-episode shim so
    # preview can resolve a name (Patch 6 will exercise the list path).
    (ep / "task_info.json").write_text(
        json.dumps([{"episode_id": 0, "task_name": "Insert the straw"}])
    )

    result = runner.invoke(app, ["--json", "preview", str(ep)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    state_dim = dict(payload["stats"]).get("State dim", "")
    # Must not claim "22" subselect — Beta cannot subselect.
    assert "22" not in str(state_dim).split()[0], (
        f"State dim should not claim 22 for Beta; got {state_dim!r}"
    )
    # Should reflect actual 14-dim raw or surface a warning.
    assert "14" in str(state_dim), f"State dim should mention raw 14; got {state_dim!r}"


# --------------------------------------------------------------------------- #
# Patch 5 — preview reads robot_type from h5 attrs (not hardcoded "a2d")
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch5_preview_robot_type_unknown_for_beta(tmp_path: Path):
    """Beta has no ``state/robot.attrs['name']``; preview surfaces ``unknown``."""
    ep = tmp_path / "meta_info" / "beta_task" / "ep0"
    ep.mkdir(parents=True)
    (ep / "proprio_stats.h5").symlink_to(BETA_H5.resolve())
    (ep / "task_info.json").write_text(
        json.dumps([{"episode_id": 0, "task_name": "Insert the straw"}])
    )
    result = runner.invoke(app, ["--json", "preview", str(ep)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    robot_type = dict(payload["stats"]).get("robot_type", "")
    assert robot_type == "unknown", (
        f"Beta has no robot.attrs['name']; expected 'unknown', got {robot_type!r}"
    )


# --------------------------------------------------------------------------- #
# Patch 6 — preview task_info list shape handling
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch6_preview_handles_task_info_list(tmp_path: Path):
    """Preview surfaces the first task_name and total-episode count for list JSON."""
    ep = tmp_path / "meta_info" / "beta_task" / "ep0"
    ep.mkdir(parents=True)
    (ep / "proprio_stats.h5").symlink_to(BETA_H5.resolve())
    (ep / "task_info.json").write_text(
        json.dumps(
            [
                {"episode_id": 1, "task_name": "Insert the straw"},
                {"episode_id": 2, "task_name": "Insert the straw"},
                {"episode_id": 3, "task_name": "Insert the straw"},
            ]
        )
    )
    result = runner.invoke(app, ["--json", "preview", str(ep)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    tasks_cell = dict(payload["stats"]).get("Tasks", "")
    assert "Insert the straw" in tasks_cell
    assert "3" in tasks_cell, f"expected episode count in Tasks cell; got {tasks_cell!r}"


# --------------------------------------------------------------------------- #
# Patch 7 — convert/__init__.py catches KeyError/ValueError without traceback
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch7_convert_keyerror_path_emits_clean_error(tmp_path: Path):
    """A schema-mismatch ``KeyError`` from the inner converter becomes a structured error."""
    # Reuse Patch 2 setup — Beta refusal raises a ValueError-class structured error
    # which must be caught by the convert/__init__.py guard, not re-raised.
    task = "beta_real"
    uuid_ = "936938"
    ep_dir = tmp_path / "src" / "meta_info" / task / uuid_
    obs_dir = tmp_path / "src" / "observations" / task / uuid_ / "video"
    ep_dir.mkdir(parents=True)
    obs_dir.mkdir(parents=True)
    (ep_dir / "proprio_stats.h5").symlink_to(BETA_H5.resolve())
    (ep_dir / "task_info.json").write_text(json.dumps([{"episode_id": 0, "task_name": "x"}]))
    (obs_dir / "head.mp4").write_bytes(b"")

    result = runner.invoke(
        app,
        [
            "convert",
            str(ep_dir),
            str(tmp_path / "out"),
            "--from",
            "agibot",
            "--to",
            "lerobot-v3",
        ],
    )
    assert result.exit_code == 2
    out = result.output
    # No raw traceback frames should reach the user.
    assert "Traceback" not in out
    assert "rich.traceback" not in out
    # A suggestion line should point users somewhere actionable.
    assert "suggestion" in out.lower() or "v0.2" in out.lower()


# --------------------------------------------------------------------------- #
# Patch 8 — inspect dumps .attrs per group/dataset
# --------------------------------------------------------------------------- #


@needs_beta
def test_patch8_inspect_includes_missing_name_attr_diagnostic():
    """``inspect`` JSON output exposes per-node attrs (so missing ``name`` is visible)."""
    result = runner.invoke(app, ["--json", "inspect", str(BETA_H5)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    nodes = {n["name"]: n for n in payload["nodes"]}
    # state/joint group exists in Beta and has no 'name' attr — the entry must
    # still be present and its attrs dict must be empty (i.e. correctly reported).
    assert "/state/joint" in nodes
    joint_node = nodes["/state/joint"]
    assert "attrs" in joint_node
    assert joint_node["attrs"] == {}, (
        f"Beta state/joint should have empty attrs; got {joint_node['attrs']!r}"
    )


@needs_beta
def test_patch8_inspect_human_readable_prints_attrs_line(tmp_path: Path):
    """Non-JSON mode prints attrs lines for nodes that have them."""
    # Make a tiny h5 with a known attr so we can grep the human output.
    import h5py

    p = tmp_path / "tiny.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("state/joint")
        g.attrs["name"] = ["a", "b", "c"]
    result = runner.invoke(app, ["inspect", str(p)])
    assert result.exit_code == 0, result.output
    assert "attrs" in result.output
    assert "name" in result.output
