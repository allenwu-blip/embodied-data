from pathlib import Path

import av
import h5py
import numpy as np

from embodied_data.validate._report import CheckResult


def _find_h5(path: Path) -> Path | None:
    direct = path / "proprio_states.h5"
    if direct.exists():
        return direct
    matches = list(path.glob("**/proprio_states.h5"))
    return matches[0] if matches else None


def _find_videos(path: Path) -> list[Path]:
    direct = path / "video"
    if direct.is_dir():
        return sorted(direct.glob("*.mp4"))
    matches = sorted(path.glob("**/video/*.mp4"))
    if matches:
        return matches
    # AgiBot release layout splits proprio (meta_info/<task>/<uuid>/) from
    # videos (observations/<task>/<uuid>/video/). When the user points at the
    # meta_info episode dir, walk up and try the sibling observations tree.
    parts = path.resolve().parts
    if "meta_info" in parts:
        idx = parts.index("meta_info")
        sibling = Path(*parts[:idx], "observations", *parts[idx + 1 :], "video")
        if sibling.is_dir():
            return sorted(sibling.glob("*.mp4"))
    return []


def check_fps(videos: list[Path]) -> CheckResult:
    if not videos:
        return CheckResult("fps consistency", "SKIP", "no videos found under */video/")
    rates: dict[float, list[str]] = {}
    for v in videos:
        try:
            with av.open(str(v)) as c:
                rate = float(c.streams.video[0].average_rate)
        except Exception as e:  # noqa: BLE001
            return CheckResult("fps consistency", "FAIL", f"{v.name}: cannot read ({e})")
        rates.setdefault(rate, []).append(v.name)
    if len(rates) == 1:
        rate = next(iter(rates))
        return CheckResult(
            "fps consistency",
            "PASS",
            f"{rate:g}fps across {len(videos)} videos (no info.json on agibot side)",
        )
    return CheckResult(
        "fps consistency",
        "FAIL",
        "rate mismatch across mp4s: " + "; ".join(f"{r}={len(n)}" for r, n in rates.items()),
    )


def check_timestamp(h5_path: Path | None, videos: list[Path]) -> CheckResult:
    if h5_path is None:
        return CheckResult("timestamp monotonicity", "SKIP", "no proprio_states.h5 found")
    with h5py.File(h5_path, "r") as f:
        if "timestamp" not in f:
            return CheckResult("timestamp monotonicity", "FAIL", "/timestamp dataset missing")
        ts = f["timestamp"][...]
    n = len(ts)
    diffs = np.diff(ts)
    mono = bool(np.all(diffs > 0)) if n > 1 else True
    if not mono:
        n_neg = int(np.sum(diffs <= 0))
        return CheckResult(
            "timestamp monotonicity",
            "FAIL",
            f"{n_neg} non-positive deltas in /timestamp ({n} samples) — issue #2689 footgun",
        )
    if n < 2 or not videos:
        return CheckResult("timestamp monotonicity", "PASS", f"{n} samples, strict mono")
    implied_rate = 1.0 / float(diffs.mean())
    try:
        with av.open(str(videos[0])) as c:
            video_rate = float(c.streams.video[0].average_rate)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "timestamp monotonicity",
            "WARN",
            f"mono OK ({n} samples, ~{implied_rate:.1f}Hz); video rate unknown ({e})",
        )
    if abs(implied_rate - video_rate) <= 0.5:
        return CheckResult(
            "timestamp monotonicity",
            "PASS",
            f"{n} samples, strict mono, ~{implied_rate:.2f}Hz matches video {video_rate:g}fps",
        )
    return CheckResult(
        "timestamp monotonicity",
        "WARN",
        (
            f"mono OK but /timestamp implies {implied_rate:.2f}Hz vs video {video_rate:g}fps "
            f"(known upstream bug — use row index, not timestamp value)"
        ),
    )


def check_action_dim(h5_path: Path | None) -> CheckResult:
    if h5_path is None:
        return CheckResult("action-dim consistency", "SKIP", "no proprio_states.h5 found")
    with h5py.File(h5_path, "r") as f:
        try:
            sj = f["state/joint/position"]
            aj = f["action/joint/position"]
        except KeyError as e:
            return CheckResult("action-dim consistency", "FAIL", f"missing dataset: {e}")
        s_shape = tuple(sj.shape)
        a_shape = tuple(aj.shape)
    if s_shape != a_shape:
        return CheckResult(
            "action-dim consistency",
            "FAIL",
            f"state/joint/position {s_shape} != action/joint/position {a_shape}",
        )
    if len(s_shape) != 2:
        return CheckResult(
            "action-dim consistency", "FAIL", f"expected 2-D joint arrays, got {s_shape}"
        )
    return CheckResult(
        "action-dim consistency",
        "PASS",
        f"{s_shape[0]} rows x dim={s_shape[1]} (state == action raw shape)",
    )


def check_alignment(h5_path: Path | None, videos: list[Path]) -> CheckResult:
    if h5_path is None or not videos:
        return CheckResult("frame-video alignment", "SKIP", "missing h5 or videos")
    with h5py.File(h5_path, "r") as f:
        if "timestamp" not in f:
            return CheckResult("frame-video alignment", "FAIL", "/timestamp dataset missing")
        n_rows = len(f["timestamp"])
    counts: dict[int, list[str]] = {}
    for v in videos:
        try:
            with av.open(str(v)) as c:
                frames = c.streams.video[0].frames
        except Exception as e:  # noqa: BLE001
            return CheckResult("frame-video alignment", "FAIL", f"{v.name}: {e}")
        counts.setdefault(frames, []).append(v.name)
    if len(counts) > 1:
        return CheckResult(
            "frame-video alignment",
            "FAIL",
            "video frame counts differ: " + "; ".join(f"{k}={len(v)}" for k, v in counts.items()),
        )
    n_frames = next(iter(counts))
    if n_rows != n_frames:
        return CheckResult(
            "frame-video alignment",
            "FAIL",
            f"h5 rows={n_rows} but each video has {n_frames} frames (#149 footgun)",
        )
    return CheckResult(
        "frame-video alignment",
        "PASS",
        f"{n_rows} h5 rows match {n_frames} frames across {len(videos)} videos",
    )


def validate(path: Path) -> list[CheckResult]:
    h5_path = _find_h5(path)
    videos = _find_videos(path)
    return [
        check_fps(videos),
        check_timestamp(h5_path, videos),
        check_action_dim(h5_path),
        check_alignment(h5_path, videos),
    ]
