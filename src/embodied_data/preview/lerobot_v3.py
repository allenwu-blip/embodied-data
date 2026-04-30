from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

# Indices of the 22 joints kept by the official AgiBot converter — unused here;
# kept commented as documentation for downstream readers.

Stat = tuple[str, str]


def _load_info(path: Path) -> dict:
    info_path = path / "meta" / "info.json"
    raw = info_path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"empty info.json at {info_path}")
    return json.loads(raw)


def _list_episode_parquets(path: Path) -> list[Path]:
    root = path / "meta" / "episodes"
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.parquet"))


def _load_tasks(path: Path) -> list[str]:
    """Return task strings. v3 uses tasks.parquet; v2.x used tasks.jsonl."""
    pq_path = path / "meta" / "tasks.parquet"
    if pq_path.is_file():
        table = pq.read_table(pq_path)
        # `task` is the dataframe index in v3; pandas writes index as a column.
        for col in ("task", "__index_level_0__"):
            if col in table.column_names:
                return [str(v) for v in table.column(col).to_pylist()]
        # Fall back: if shape is (N, 1) and the only column is task_index, no strings.
        return []
    jsonl_path = path / "meta" / "tasks.jsonl"
    if jsonl_path.is_file():
        out: list[str] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(str(obj.get("task", "")))
        return out
    return []


def _episode_lengths(parquets: list[Path], n: int) -> tuple[int, list[int]]:
    """Return (total_episodes_seen_across_files, lengths of first n episodes)."""
    lengths: list[int] = []
    total = 0
    for p in parquets:
        if len(lengths) >= n:
            break
        table = pq.read_table(p, columns=["length"])
        col = table.column("length").to_pylist()
        total += len(col)
        for length in col:
            if len(lengths) >= n:
                break
            lengths.append(int(length))
    return total, lengths


def collect_lerobot_v3_stats(path: Path, n: int) -> tuple[list[Stat], str]:
    info = _load_info(path)
    fps = int(info.get("fps") or 0)
    if fps <= 0:
        raise ValueError("info.json missing positive fps")
    total_episodes_meta = int(info.get("total_episodes") or 0)
    robot_type = info.get("robot_type") or "unknown"
    features: dict = info.get("features") or {}

    state_dim = _shape_first(features.get("observation.state"))
    action_dim = _shape_first(features.get("action"))
    cameras = sorted(k for k in features if k.startswith("observation.images."))

    parquets = _list_episode_parquets(path)
    seen_total, lengths = _episode_lengths(parquets, n)
    # Prefer info.total_episodes if metadata files are absent; otherwise use the
    # count we walked over the parquet to report a "seen" total.
    total_episodes = max(total_episodes_meta, seen_total)
    sampled = len(lengths)
    total_frames = sum(lengths)
    duration_s = total_frames / fps if fps else 0.0

    tasks = _load_tasks(path)
    task_summary = _format_tasks(tasks)

    header = ""
    if sampled and total_episodes and sampled < total_episodes:
        header = f"showing first {sampled} of {total_episodes} episodes"

    stats: list[Stat] = [
        ("Format", "lerobot-v3"),
        ("Episodes", f"{sampled} sampled / {total_episodes} total"),
        ("Total frames", str(total_frames)),
        ("Total duration", f"{duration_s:.1f}s"),
        ("fps", str(fps)),
        ("robot_type", str(robot_type)),
        ("State dim", str(state_dim) if state_dim is not None else "?"),
        ("Action dim", str(action_dim) if action_dim is not None else "?"),
        ("Cameras", ", ".join(cameras) if cameras else "(none)"),
        ("Tasks", task_summary),
    ]
    return stats, header


def _shape_first(feat: dict | None) -> int | None:
    if not feat:
        return None
    shape = feat.get("shape")
    if not shape:
        return None
    return int(shape[0])


def _format_tasks(tasks: list[str]) -> str:
    if not tasks:
        return "0"
    head = tasks[:3]
    quoted = ", ".join(f'"{t}"' for t in head)
    return f"{len(tasks)}: {quoted}" if len(tasks) <= 3 else f"{len(tasks)}: {quoted}, ..."
