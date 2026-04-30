"""Tiny dev tool: dump the schema of a single .h5 or .parquet file."""

from __future__ import annotations

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


def run_inspect(*, path: Path) -> None:
    if not path.exists() or not path.is_file():
        emit_error(
            f"not a file: {path}",
            suggestion="inspect takes a single .h5 or .parquet file path",
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
