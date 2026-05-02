"""Inspect: dump the schema of a single .h5 or .parquet file (default mode),
or print a high-level summary of a LeRobot v3 dataset directory (--summary)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from embodied_data._emit import emit_error, emit_json, get_console
from embodied_data._state import state

console = get_console()


def _inspect_h5(path: Path) -> dict[str, Any]:
    import h5py

    nodes: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:

        def _attrs(obj: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in obj.attrs.items():
                if hasattr(v, "tolist"):
                    out[k] = v.tolist()
                else:
                    out[k] = str(v)
            return out

        def _walk(name: str, obj: Any) -> None:
            entry: dict[str, Any] = {"name": "/" + name, "attrs": _attrs(obj)}
            if isinstance(obj, h5py.Dataset):
                entry["kind"] = "dataset"
                entry["shape"] = list(obj.shape)
                entry["dtype"] = str(obj.dtype)
            else:
                entry["kind"] = "group"
            nodes.append(entry)

        f.visititems(_walk)
    return {"path": str(path), "kind": "h5", "nodes": nodes}


def _inspect_parquet(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    schema = [{"name": f.name, "type": str(f.type)} for f in table.schema]
    head_rows = table.slice(0, min(3, table.num_rows)).to_pylist()
    # JSON-safe scrub: keep only primitive types in the preview.
    safe_rows = [{k: _to_json_safe(v) for k, v in row.items()} for row in head_rows]
    return {
        "path": str(path),
        "kind": "parquet",
        "schema": schema,
        "num_rows": table.num_rows,
        "head": safe_rows,
    }


_ATTR_VALUE_MAX = 80


def _truncate_attr_value(v: Any) -> Any:
    """Limit a single attr value to 80 chars in human-mode display.

    Strings longer than 80 chars get an ellipsis. Lists/tuples are truncated
    by element count once their repr passes the limit. Other types fall
    through to ``str(...)`` with the same length cap.
    """
    if isinstance(v, str):
        return v if len(v) <= _ATTR_VALUE_MAX else v[: _ATTR_VALUE_MAX - 3] + "..."
    if isinstance(v, (list, tuple)):
        rendered = repr(v)
        if len(rendered) <= _ATTR_VALUE_MAX:
            return v
        # Drop trailing elements until we fit.
        head = list(v)
        while head and len(repr(head + ["..."])) > _ATTR_VALUE_MAX:
            head.pop()
        return head + ["..."]
    s = str(v)
    return s if len(s) <= _ATTR_VALUE_MAX else s[: _ATTR_VALUE_MAX - 3] + "..."


def _to_json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_json_safe(val) for k, val in v.items()}
    return str(v)


def _human_bytes(n: int) -> str:
    """1234567 → '1.2 MB'. Power-of-1024, two-digit precision under 10."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


def _du_recursive(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _summarize_v3_dataset(path: Path) -> dict[str, Any]:
    """Read a LeRobot v3 dataset's metadata + on-disk state into a summary dict.

    Returns keys: path, fps, robot_type, total_episodes, total_frames,
    duration_seconds, state_dim, action_dim, cameras (list of
    {key, resolution, codec, fps}), disk_bytes, validate_results
    (list of {name, status, detail}), overall_status.
    """
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        emit_error(
            f"not a LeRobot v3 dataset (missing {info_path})",
            suggestion=(
                "inspect --summary expects a v3 dataset root with meta/info.json. "
                "For single .h5 / .parquet files, omit --summary."
            ),
            exit_code=2,
        )

    info = json.loads(info_path.read_text())
    features = info.get("features") or {}

    fps = float(info.get("fps") or 0)
    total_frames = int(info.get("total_frames") or 0)
    duration_seconds = (total_frames / fps) if fps > 0 else 0.0

    def _shape_dim(name: str) -> int | None:
        feat = features.get(name)
        if not isinstance(feat, dict):
            return None
        shape = feat.get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[0])
        return None

    cameras: list[dict[str, Any]] = []
    for key, feat in features.items():
        if not isinstance(feat, dict) or feat.get("dtype") != "video":
            continue
        feat_info = feat.get("info") or {}
        shape = feat.get("shape") or []
        # shape is [height, width, channels]
        res = (
            f"{int(shape[1])}x{int(shape[0])}"
            if len(shape) >= 2 and shape[0] and shape[1]
            else "unknown"
        )
        cameras.append(
            {
                "key": key,
                "resolution": res,
                "codec": str(feat_info.get("video.codec") or "unknown"),
                "fps": float(feat_info.get("video.fps") or fps),
            }
        )

    # Five-check validate (lerobot-v3 path) — surface PASS/FAIL/SKIP status only,
    # not full re-validation report.
    from embodied_data.validate import lerobot_v3 as v3

    check_fns = [
        v3.check_schema_conformance,
        v3.check_fps,
        v3.check_timestamp,
        v3.check_action_dim,
        v3.check_alignment,
    ]
    validate_results: list[dict[str, str]] = []
    for fn in check_fns:
        try:
            r = fn(path, info)
            validate_results.append({"name": r.name, "status": r.status, "detail": r.detail})
        except Exception as exc:  # noqa: BLE001 — surface any check crash as FAIL
            validate_results.append(
                {"name": fn.__name__, "status": "FAIL", "detail": f"check raised: {exc}"}
            )

    overall = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in validate_results)
        else ("WARN" if any(r["status"] == "WARN" for r in validate_results) else "PASS")
    )

    return {
        "path": str(path),
        "fps": fps,
        "robot_type": str(info.get("robot_type") or "unknown"),
        "total_episodes": int(info.get("total_episodes") or 0),
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 3),
        "state_dim": _shape_dim("observation.state"),
        "action_dim": _shape_dim("action"),
        "cameras": cameras,
        "disk_bytes": _du_recursive(path),
        "validate_results": validate_results,
        "overall_status": overall,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    """Render the summary dict as Rich tables + overall PASS/FAIL footer."""
    from rich.table import Table

    overview = Table(title=f"Dataset summary: {summary['path']}", show_header=True)
    overview.add_column("Field", style="bold")
    overview.add_column("Value")
    duration = summary["duration_seconds"]
    minutes = int(duration // 60)
    seconds = duration - minutes * 60
    overview.add_row("Robot type", str(summary["robot_type"]))
    overview.add_row("FPS", f"{summary['fps']:g}")
    overview.add_row("Episodes", str(summary["total_episodes"]))
    overview.add_row("Frames", str(summary["total_frames"]))
    overview.add_row("Duration", f"{minutes}m {seconds:.1f}s ({duration:.1f}s)")
    overview.add_row("State dim", str(summary["state_dim"]))
    overview.add_row("Action dim", str(summary["action_dim"]))
    overview.add_row("Cameras", str(len(summary["cameras"])))
    overview.add_row("Disk size", _human_bytes(summary["disk_bytes"]))
    console.print(overview)

    if summary["cameras"]:
        cams = Table(title="Cameras", show_header=True)
        cams.add_column("Key")
        cams.add_column("Resolution")
        cams.add_column("Codec")
        cams.add_column("FPS")
        for cam in summary["cameras"]:
            cams.add_row(cam["key"], cam["resolution"], cam["codec"], f"{cam['fps']:g}")
        console.print(cams)

    checks = Table(title="Validation checks", show_header=True)
    checks.add_column("Check")
    checks.add_column("Status")
    checks.add_column("Detail")
    status_style = {"PASS": "green", "FAIL": "red", "WARN": "yellow", "SKIP": "dim"}
    for r in summary["validate_results"]:
        style = status_style.get(r["status"], "")
        status_cell = f"[{style}]{r['status']}[/{style}]" if style else r["status"]
        checks.add_row(r["name"], status_cell, r["detail"])
    console.print(checks)

    overall = summary["overall_status"]
    color = status_style.get(overall, "")
    if color:
        console.print(f"Overall: [{color}]{overall}[/{color}]")
    else:
        console.print(f"Overall: {overall}")


def run_inspect(*, path: Path, summary: bool = False) -> None:
    if summary:
        if not path.exists() or not path.is_dir():
            emit_error(
                f"not a directory: {path}",
                suggestion="inspect --summary takes a LeRobot v3 dataset directory path",
                exit_code=2,
            )
        payload = _summarize_v3_dataset(path)
        if state.json_output:
            emit_json(payload)
        else:
            _print_summary(payload)
        if payload["overall_status"] == "FAIL":
            import typer

            raise typer.Exit(code=1)
        return

    if not path.exists() or not path.is_file():
        emit_error(
            f"not a file: {path}",
            suggestion=(
                "inspect takes a single .h5 or .parquet file path "
                "(or use --summary for a dataset directory)"
            ),
            exit_code=2,
        )

    suf = path.suffix.lower()
    if suf in (".h5", ".hdf5"):
        payload = _inspect_h5(path)
    elif suf == ".parquet":
        payload = _inspect_parquet(path)
    else:
        emit_error(
            f"unsupported file type: {path.suffix}",
            suggestion="inspect supports .h5 / .hdf5 and .parquet only",
            exit_code=2,
        )
        return  # unreachable

    if state.json_output:
        emit_json(payload)
        return

    console.print(f"Inspect: {path}  (kind: {payload['kind']})")
    if payload["kind"] == "h5":
        for n in payload["nodes"]:
            if n["kind"] == "dataset":
                console.print(f"  {n['name']}  shape={n['shape']}  dtype={n['dtype']}")
            else:
                console.print(f"  {n['name']}/  (group)")
            if n["attrs"]:
                # Truncate each attr value to 80 chars so a multi-MB blob doesn't
                # blow up the dump. Per-key truncation (not Rich line-wrap) so the
                # raw value cannot be reconstructed by stripping newlines.
                truncated = {k: _truncate_attr_value(v) for k, v in n["attrs"].items()}
                console.print(f"      attrs={truncated}")
    else:
        console.print(f"  rows: {payload['num_rows']}")
        for col in payload["schema"]:
            console.print(f"  {col['name']}: {col['type']}")
        if payload["head"]:
            console.print("  head (first 3 rows):")
            for row in payload["head"]:
                console.print(f"    {row}")
