# AgiBot Beta — HDF5 Schema Reference

Real-hardware AgiBot capture. `embodied-data` v0.2 supports forward
conversion to LeRobot v3 via
`convert_agibot_beta_to_lerobot_v3` (single-episode) and
`convert_agibot_beta_batch` (multi-episode + `--resume` / `--workers`).
v0.3+ adds `observation.images.head_color` video re-encoding when
`<episode_dir>/videos/head_color.mp4` is present (see §10).

> **Alpha applies too.** `agibot-world/AgiBotWorld-Alpha` and
> `agibot-world/AgiBotWorld-Beta` share schemas (verified 2026-04-30 — see
> [`overview.md`](./overview.md) for the head-to-head diff). The Beta
> converter handles Alpha unchanged; the dispatcher routes Alpha-named paths
> through the Beta path automatically.

---

## 1. Sample fixture

Two episodes ship under `data/agibot_beta_sample/675/`:

- **`936938/proprio_stats.h5`** — proprio-only (Beta task 675 "Insert the
  straw", 1090 frames). Acquired by stream-extracting one h5 from
  `proprio_stats/<chunk>.tar` per Sprint 3 B.alpha-hunter, ~1.2 MB
  streamed of an 800 MB tar. Used as the v0.2 legacy / no-video fixture.
- **`882736/{proprio_stats.h5, videos/head_color.mp4}`** — proprio +
  head_color video (879 frames @ 30 fps, 640×480, av1 upstream). The
  video is stream-extracted from the 36 GB
  `observations/675/880749-912853.tar` via HTTP Range requests against
  HF (~8 MB downloaded). Used as the v0.3 video fixture.

Reproducible acquisition is in `scripts/fetch_beta_video_fixture.py`:

```bash
huggingface-cli login                       # gated dataset access
uv run python scripts/fetch_beta_video_fixture.py
```

Equivalent one-liner for the head_color video alone (HTTP Range against
the upstream tar — no full-tar download needed):

```python
import urllib.request, pathlib
from huggingface_hub import hf_hub_url, get_hf_file_metadata
from huggingface_hub.utils import build_hf_headers

url = hf_hub_url(
    "agibot-world/AgiBotWorld-Beta",
    "observations/675/880749-912853.tar",
    repo_type="dataset",
)
meta = get_hf_file_metadata(url)
# Walk tar headers (each 512 B) to find <ep>/videos/head_color.mp4
# byte range, then Range-fetch just that slice (~8 MB, not 36 GB).
```

## 2. Filename

`proprio_stats.h5` — note the dropped `e` from "states". Upstream README
got it right; the v0.1 sim converter was hard-coded to `proprio_states.h5`
and silently failed on Beta. Fixed in v0.1.1 via `_agibot_paths.PROPRIO_GLOB`.

## 3. Top-level layout

The Beta task-dataset root contains one `task_info_<task>.json` per task at
the root level (one big list of episode dicts) plus a `<task>/` subtree per
task with one `<episode_id>/` subdir per episode:

```
agibot-world/AgiBotWorld-Beta/        (HF; tarred upstream — see Sprint 3)
├── task_info_<task>.json             (list of N episode dicts)
├── <task>/
│   ├── <episode_id>/
│   │   └── proprio_stats.h5
│   └── ...
├── proprio_stats/<chunk>.tar         (upstream packaging — extracted on use)
└── observations/<task>/<chunk>.tar   (videos — out of scope for v0.2)
```

After extraction (the layout `embodied-data` actually consumes):

```
data/agibot_beta_sample/
├── task_info_675.json                (list of 399 dicts for task 675)
└── 675/
    ├── 882736/                       (v0.3 fixture: proprio + head_color video)
    │   ├── proprio_stats.h5
    │   └── videos/
    │       └── head_color.mp4
    └── 936938/                       (v0.2 fixture: proprio-only)
        └── proprio_stats.h5
```

## 4. h5 group structure

`f.visit(print)` produces 53 paths. The headline shapes (sample episode 1090 frames):

| Path | Shape | Dtype | Used by v0.2 converter |
|---|---|---|---|
| `state/joint/position` | `(1090, 14)` | float64 | yes (14 → first 14 of 20-dim observation.state) |
| `state/joint.attrs["name"]` | — | — | **missing** (v0.2 uses hardcoded `JOINT_14_BETA` best-guess names) |
| `state/effector/position` | `(1090, 2)` | float64 | yes (2 → idx 14-15 of 20) |
| `state/head/position` | `(1090, 2)` | float64 | yes (2 → idx 16-17) |
| `state/waist/position` | `(1090, 2)` | float64 | yes (2 → idx 18-19) |
| `state/end/{position,orientation,velocity,wrench,angular}` | various | float64 | no (v0.2 — out of scope; 32-dim end-pose is v0.2.1 candidate) |
| `state/robot/{position,orientation,*_drift}` | various | float64 | no (mobile-base out of v0.2 scope) |
| `state/joint/{velocity,effort,current_value}` | various | float64 | no (v0.2) |
| `action/joint/{position,velocity,effort,index}` | matching | float64 | first-difference of state, see §5 |
| `action/effector/{force,position,index}` | matching | float64 | no (v0.2) |
| `action/end/...`, `action/robot/...`, `action/head/...`, `action/waist/...` | matching | float64 | no (v0.2) |
| `timestamp` | `(1090,)` | **int64** | **discarded**, recomputed as `frame_index/30` (see §6) |

## 5. Action recipe (first-difference of state)

`embodied-data` v0.2 does not read `action/joint/position` directly because
sim's identity (action == state in DigitalWorld) makes the raw column a
weak signal. Real Beta captures may diverge but we did not verify per-frame
equality on the larger Beta corpus, so v0.2 keeps the conservative recipe:
`action[i] = state[i+1] − state[i]` for `i ∈ [0, N-2]`, with the last
delta repeated to length N. This matches the upstream sim
`convert_to_lerobot.py:293-302` recipe.

When real Beta `action/joint/position` is empirically a useful signal in
its own right (we'd need to measure on multi-episode samples to tell), the
recipe can be revisited in a v0.2.x patch without breaking the on-disk
contract — `observation.state` and `action` are both 20-dim float32, so
swapping the action source is internal.

## 6. Timestamp handling

Beta stores `int64` ns Unix-epoch timestamps. Sample episode 936938 spans
1740109484562531000 → 1740109520909683000 (≈36.3 s). v0.2 **discards** this
column and recomputes `timestamp = frame_index / 30.0` per the LeRobot v3
invariant (the read-time tolerance check in `decode_video_frames` requires
this). Side effect: any frame-rate jitter or dropped frames invisible
post-conversion. Acceptable for v0.2; revisit in v0.3 with a per-frame
preserved column under `auxiliary.timestamp_raw` if users need it.

## 7. Joint name guess (provenance + revision path)

`state/joint.attrs["name"]` is missing on Beta (and Alpha). v0.2 uses
`JOINT_14_BETA = [arm_l_j1..7, arm_r_j1..7]` as a best guess based on the
URDF naming convention sim DigitalWorld uses for arms. If a user reports
their Beta task has different ordering, v0.2.1 can add a
`--joint-names <file.json>` override flag without breaking the existing
constant.

## 8. Sparse `*/index` companions (currently dropped)

Beta action subgroups carry an `index` array sized differently from frame
count (e.g., `action/joint/index` 1046 vs 1090). Per upstream README these
mark "when the control source is actually sending signals" — sparse mask
metadata. LeRobot v3 has no canonical slot. v0.2 drops these silently.
v0.3 candidate: surface as `auxiliary.<group>.mask` features with frame-
aligned 1090-length boolean arrays.

## 9. `task_info_<task>.json` shape

A JSON **list of N episode dicts** at the task-dataset root, where N is the
number of episodes in this task. Each dict has at minimum:

```jsonc
{
  "episode_id": 936938,
  "task_name": "Insert the straw",
  "init_scene_text": "...",
  "label_info": {...}
}
```

`embodied-data` v0.2's `_resolve_task_name_from_file` returns the first
entry's `task_name` (Beta tasks have a single canonical name shared by all
episodes — by design). If that proves insufficient (task descriptions vary
per-episode), the fallback to indexed lookup by `episode_id` is a v0.2.1
patch.

## 10. Video — `observation.images.head_color` (v0.3+)

Beta upstream packages videos as multi-GB tars at
`observations/<task>/<chunk>.tar` with the per-episode layout
`<episode_id>/videos/{head_color,fisheye_left,fisheye_right,hand_left,hand_right,back_left,back_right}.mp4`.
Codec is typically av1 @ 30 fps.

`embodied-data` v0.3+ ingests `head_color.mp4` only (other cameras are
v0.3.1 candidates):

- Single-episode: `convert_agibot_beta_to_lerobot_v3` looks for
  `<src>/videos/head_color.mp4`. If present, it's re-encoded through
  the LeRobot v3 video contract — h264, `bf=0`, `g=2`, `yuv420p`,
  monotonic PTS aligned to `frame_index` — and emitted at
  `<dst>/videos/observation.images.head_color/chunk-000/file-000.mp4`.
- Batch: `convert_agibot_beta_batch` is all-or-nothing per dataset.
  If any episode has the upstream mp4, the dataset declares the
  `observation.images.head_color` feature; episodes missing it are
  logged to `.beta_batch_errors.jsonl` and skipped. If no episode
  has video, output is proprio-only (legacy v0.2).

Output `info.features[observation.images.head_color]`:

```json
{
  "dtype": "video",
  "shape": [height, width, 3],
  "names": ["height", "width", "channels"],
  "info": {
    "video.fps": 30.0,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.height": <H>,
    "video.width": <W>,
    "video.channels": 3,
    "video.is_depth_map": false
  }
}
```

Per-episode meta gains `videos/observation.images.head_color/{chunk_index, file_index, from_timestamp, to_timestamp}` columns. `info.json.video_path` flips from `null` to the LeRobot v3 standard template.

Hard-fail before re-encode if upstream video frame count diverges
from proprio frame count by more than 1 frame.

## 11. Out-of-scope for v0.3.0

- Multi-camera (`fisheye_left/right`, `hand_left/right`,
  `back_left/right`) — v0.3.1
- `state/end/*` flattening into `observation.state.end_pose` (32-dim) —
  v0.3.3 candidate
- `action/{end,robot,head,waist}/*` (only `joint` is captured in
  `observation.state`)
- Sparse `*/index` companions (see §8) — v0.3.2 candidate
- Reverse `lerobot-v3 → agibot-beta`
- Cross-embodiment retargeting (DigitalWorld 22-dim → Beta 20-dim or
  vice versa)
