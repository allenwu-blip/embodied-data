import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from embodied_data.validate._report import CheckResult


def _load_info(path: Path) -> dict | None:
    info_p = path / "meta" / "info.json"
    if not info_p.exists():
        return None
    with info_p.open() as f:
        return json.load(f)


def _list_videos(path: Path) -> list[Path]:
    vd = path / "videos"
    if not vd.is_dir():
        return []
    return sorted(vd.glob("**/*.mp4"))


def _list_data_parquets(path: Path) -> list[Path]:
    dd = path / "data"
    if not dd.is_dir():
        return []
    return sorted(dd.glob("**/*.parquet"))


def check_fps(path: Path, info: dict) -> CheckResult:
    declared = info.get("fps")
    if declared is None:
        return CheckResult("fps consistency", "FAIL", "info.fps missing in meta/info.json")
    videos = _list_videos(path)
    if not videos:
        return CheckResult("fps consistency", "SKIP", "no videos under videos/")
    mismatches: list[str] = []
    rates: set[float] = set()
    for v in videos:
        try:
            with av.open(str(v)) as c:
                rate = float(c.streams.video[0].average_rate)
        except Exception as e:  # noqa: BLE001
            return CheckResult("fps consistency", "FAIL", f"{v.name}: cannot read ({e})")
        rates.add(rate)
        if abs(rate - float(declared)) > 1e-3:
            mismatches.append(f"{v.relative_to(path)}={rate:g}")
    if mismatches:
        return CheckResult(
            "fps consistency",
            "FAIL",
            f"info.fps={declared} but: " + ", ".join(mismatches[:5]),
        )
    only = next(iter(rates))
    return CheckResult(
        "fps consistency",
        "PASS",
        f"{only:g}fps across {len(videos)} video(s), matches info.fps={declared}",
    )


def check_timestamp(path: Path, info: dict) -> CheckResult:
    parquets = _list_data_parquets(path)
    if not parquets:
        return CheckResult("timestamp monotonicity", "SKIP", "no data/*.parquet")
    fps = float(info.get("fps") or 0)
    total_rows = 0
    spikes: list[str] = []
    for p in parquets:
        try:
            tbl = pq.read_table(p, columns=["timestamp", "episode_index"])
        except Exception as e:  # noqa: BLE001
            return CheckResult("timestamp monotonicity", "FAIL", f"{p.name}: {e}")
        ts = tbl.column("timestamp").to_numpy()
        ep = tbl.column("episode_index").to_numpy()
        total_rows += len(ts)
        for ep_id in np.unique(ep):
            mask = ep == ep_id
            seg = ts[mask]
            if len(seg) < 2:
                continue
            diffs = np.diff(seg)
            if np.any(diffs < 0):
                # issue #2689 sudden negative delta
                idx = int(np.argmin(diffs))
                spikes.append(f"ep={int(ep_id)} idx={idx} dt={float(diffs[idx]):.6f}")
            elif fps > 0 and np.any(diffs > 5.0 / fps):
                # large positive jump (>5 frames) — also a #2689 symptom
                idx = int(np.argmax(diffs))
                spikes.append(f"ep={int(ep_id)} idx={idx} jump={float(diffs[idx]):.4f}s")
    if spikes:
        return CheckResult(
            "timestamp monotonicity",
            "FAIL",
            f"{len(spikes)} non-monotone/spike(s); first: {spikes[0]} (issue #2689)",
        )
    return CheckResult(
        "timestamp monotonicity",
        "PASS",
        f"{total_rows} rows, strict mono per episode, no spikes",
    )


def _feature_dim(info: dict, key: str) -> int | None:
    feats = info.get("features", {})
    feat = feats.get(key)
    if feat is None:
        return None
    shape = feat.get("shape")
    if not shape:
        return None
    return int(shape[0])


def check_action_dim(path: Path, info: dict) -> CheckResult:
    parquets = _list_data_parquets(path)
    if not parquets:
        return CheckResult("action-dim consistency", "SKIP", "no data/*.parquet")
    declared_a = _feature_dim(info, "action")
    declared_s = _feature_dim(info, "observation.state")
    seen_a: set[int] = set()
    seen_s: set[int] = set()
    total = 0
    for p in parquets:
        cols = []
        schema = pq.read_schema(p)
        names = set(schema.names)
        for c in ("action", "observation.state"):
            if c in names:
                cols.append(c)
        if not cols:
            continue
        tbl = pq.read_table(p, columns=cols)
        total += tbl.num_rows
        if "action" in cols:
            arr = tbl.column("action").to_pylist()
            seen_a.update(len(x) for x in arr if x is not None)
        if "observation.state" in cols:
            arr = tbl.column("observation.state").to_pylist()
            seen_s.update(len(x) for x in arr if x is not None)
    issues: list[str] = []
    if len(seen_a) > 1:
        issues.append(f"action dims vary: {sorted(seen_a)}")
    if len(seen_s) > 1:
        issues.append(f"observation.state dims vary: {sorted(seen_s)}")
    if declared_a is not None and seen_a and declared_a not in seen_a:
        issues.append(f"action dim {sorted(seen_a)} != info.features.action.shape[0]={declared_a}")
    if declared_s is not None and seen_s and declared_s not in seen_s:
        issues.append(
            f"state dim {sorted(seen_s)} != info.features.observation.state.shape[0]={declared_s}"
        )
    if issues:
        return CheckResult("action-dim consistency", "FAIL", "; ".join(issues))
    a = next(iter(seen_a)) if seen_a else "?"
    s = next(iter(seen_s)) if seen_s else "?"
    return CheckResult(
        "action-dim consistency",
        "PASS",
        f"{total} rows, action dim={a}, state dim={s}, matches info.features",
    )


def check_alignment(path: Path, info: dict) -> CheckResult:
    parquets = _list_data_parquets(path)
    if not parquets:
        return CheckResult("frame-video alignment", "SKIP", "no data/*.parquet")
    fps = float(info.get("fps") or 0)
    if fps <= 0:
        return CheckResult("frame-video alignment", "SKIP", "info.fps missing/invalid")
    # Use episode metadata to map (episode -> mp4 path + from/to ts)
    ep_meta_files = sorted((path / "meta" / "episodes").glob("**/*.parquet"))
    if not ep_meta_files:
        # Fallback: row-count vs total mp4 frames if only one camera & one mp4
        videos = _list_videos(path)
        total_rows = sum(pq.read_metadata(p).num_rows for p in parquets)
        total_frames = 0
        for v in videos:
            try:
                with av.open(str(v)) as c:
                    total_frames += c.streams.video[0].frames
            except Exception as e:  # noqa: BLE001
                return CheckResult("frame-video alignment", "FAIL", f"{v.name}: {e}")
        if not videos:
            return CheckResult("frame-video alignment", "SKIP", "no videos to align")
        # one video per episode assumption
        per_cam_frames = total_frames // max(len(videos), 1)
        if total_rows == per_cam_frames:
            return CheckResult(
                "frame-video alignment",
                "PASS",
                f"{total_rows} rows match {per_cam_frames} frames/camera (no episode meta)",
            )
        return CheckResult(
            "frame-video alignment",
            "FAIL",
            f"rows={total_rows} != frames/camera={per_cam_frames} (#149 footgun)",
        )

    # Per-episode check using meta/episodes
    issues: list[str] = []
    n_eps = 0
    n_rows_total = 0
    for emf in ep_meta_files:
        ep_tbl = pq.read_table(emf)
        cols = set(ep_tbl.schema.names)
        if "length" not in cols or "episode_index" not in cols:
            return CheckResult(
                "frame-video alignment",
                "FAIL",
                f"{emf.name} missing required columns 'episode_index'/'length'",
            )
        ep_idx = ep_tbl.column("episode_index").to_pylist()
        lengths = ep_tbl.column("length").to_pylist()
        # video-key → (chunk_index, file_index, from_ts, to_ts)
        vid_keys = [
            n.split("/", 1)[1].rsplit("/", 1)[0]
            for n in cols
            if n.startswith("videos/") and n.endswith("/from_timestamp")
        ]
        for i, length in enumerate(lengths):
            n_eps += 1
            n_rows_total += int(length)
            for vk in vid_keys:
                ci = int(ep_tbl.column(f"videos/{vk}/chunk_index")[i].as_py())
                fi = int(ep_tbl.column(f"videos/{vk}/file_index")[i].as_py())
                ft = float(ep_tbl.column(f"videos/{vk}/from_timestamp")[i].as_py())
                tt = float(ep_tbl.column(f"videos/{vk}/to_timestamp")[i].as_py())
                vp = path / "videos" / vk / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
                if not vp.exists():
                    issues.append(f"ep{ep_idx[i]} {vk}: missing {vp.relative_to(path)}")
                    continue
                expected_dur = length / fps
                actual_dur = tt - ft
                if abs(actual_dur - expected_dur) > 1.5 / fps:
                    issues.append(
                        f"ep{ep_idx[i]} {vk}: rows={length} (={expected_dur:.3f}s) "
                        f"vs ts span {actual_dur:.3f}s (>1 frame off)"
                    )
    if issues:
        return CheckResult(
            "frame-video alignment",
            "FAIL",
            f"{len(issues)} misalignment(s); first: {issues[0]}",
        )
    return CheckResult(
        "frame-video alignment",
        "PASS",
        f"{n_eps} episode(s), {n_rows_total} rows ↔ video durations within ±1 frame",
    )


def check_schema_conformance(path: Path, info: dict) -> CheckResult:
    """Surface the spec violations C found in real HF datasets:
    - codebase_version mismatch (e.g. v2.1 silently mutated)
    - int fields nulled-out (chunks_size, data/video_files_size_in_mb)
    - meta/episodes parquet missing 'tasks' column (spec §3)
    - meta/tasks.parquet missing both 'task' and '__index_level_0__' columns
    """
    fails: list[str] = []
    warns: list[str] = []

    cv = info.get("codebase_version")
    if cv != "v3.0":
        fails.append(f"codebase_version={cv!r}, expected 'v3.0'")

    for k in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"):
        v = info.get(k)
        if v is None:
            warns.append(f"{k} is null (spec says int; readers fall back to defaults)")
        elif not isinstance(v, int):
            warns.append(f"{k}={v!r} not int")

    ep_meta_files = sorted((path / "meta" / "episodes").glob("**/*.parquet"))
    if ep_meta_files:
        try:
            schema = pq.read_schema(ep_meta_files[0])
            if "tasks" not in set(schema.names):
                fails.append(f"{ep_meta_files[0].name} missing required 'tasks' column (spec §3)")
        except Exception as exc:  # noqa: BLE001
            fails.append(f"cannot read {ep_meta_files[0].name}: {exc}")

    tasks_pq = path / "meta" / "tasks.parquet"
    if tasks_pq.is_file():
        try:
            tcols = set(pq.read_schema(tasks_pq).names)
            if "task" not in tcols and "__index_level_0__" not in tcols:
                fails.append(
                    f"tasks.parquet has neither 'task' nor '__index_level_0__'; got {sorted(tcols)}"
                )
        except Exception as exc:  # noqa: BLE001
            fails.append(f"cannot read tasks.parquet: {exc}")

    if fails:
        detail = "; ".join(fails)
        if warns:
            detail += f" | warns: {'; '.join(warns)}"
        return CheckResult("schema conformance", "FAIL", detail)
    if warns:
        return CheckResult("schema conformance", "WARN", "; ".join(warns))
    return CheckResult(
        "schema conformance",
        "PASS",
        f"codebase_version={cv}, info.json int fields populated, episode meta has 'tasks'",
    )


def validate(path: Path) -> list[CheckResult]:
    info = _load_info(path)
    if info is None:
        return [CheckResult("meta/info.json", "FAIL", "missing")]
    return [
        check_schema_conformance(path, info),
        check_fps(path, info),
        check_timestamp(path, info),
        check_action_dim(path, info),
        check_alignment(path, info),
    ]
