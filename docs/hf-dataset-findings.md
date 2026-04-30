# HuggingFace v3 dataset validation — Sprint 2 findings

Three real, public LeRobot v3 datasets fetched from HuggingFace, validated with
`embodied-data validate` at commit `960f725`.

- **Date**: 2026-04-30
- **Tooling**: `embodied-data` rev `960f72586d8bc68984cf514742d7d53009959d0d`
- **HF token**: not used (public-only)
- **Download budget**: per-dataset ≤500MB, total ≤1.5GB (actual: ~271MB total)
- **Storage**: `data/hf_v3_samples/<name>/` (gitignored)

Download policy was **head-only**: `meta/*` + `data/chunk-000/file-000.parquet` +
`videos/*/chunk-000/file-000.mp4`. The intent is to exercise validate on a
representative slice without paying for full corpora; the so101 case below
explicitly shows this trade-off.

---

## Dataset 1 — `lerobot/pusht`

**Why picked**: official-HF, simulated 2D push-T benchmark, single image stream
(`observation.image`, not `.images.*`), low fps (10), small (7.7MB head). Tests
the "single-camera, image-not-images" code path.

| Field | Value |
|---|---|
| `repo_id` | `lerobot/pusht` |
| `codebase_version` | `v3.0` |
| `robot_type` | `unknown` |
| `fps` | 10 |
| `total_episodes` | 206 |
| `total_frames` | 25650 |
| `total_tasks` | 1 |
| Head download | 7.68 MB |
| Camera keys | `observation.image` (singular) |

### Validate output (verbatim)

```
Validation: /Users/allenwu/embodied-data/data/hf_v3_samples/pusht  (format: lerobot-v3)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                  ┃ Status ┃ Detail                                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ fps consistency        │ PASS   │ 10fps across 1 video(s), matches info.fps=10                 │
│ timestamp monotonicity │ PASS   │ 25650 rows, strict mono per episode, no spikes               │
│ action-dim consistency │ PASS   │ 25650 rows, action dim=2, state dim=2, matches info.features │
│ frame-video alignment  │ PASS   │ 206 episode(s), 25650 rows ↔ video durations within ±1 frame │
└────────────────────────┴────────┴──────────────────────────────────────────────────────────────┘
Result: PASS
```

Exit code: **0** (PASS).

**Interpretation**: Clean v3, full 206 episodes & 25650 frames consolidated into
one parquet + one mp4; our validator passes legitimately.

**Side observation (not a fail)**: `video_files_size_in_mb=500`, but the
embodied-data spec doc (`docs/schema-lerobot-v3.md` §2) says default is **200**.
The current LeRobot writer must have changed default or this dataset was
re-encoded with a custom config. Worth a follow-up to update our spec doc.

---

## Dataset 2 — `lerobot/unitreeh1_warehouse`

**Why picked**: official-HF **humanoid** (Unitree H1), the closest substitute for
"agibot-style" since `lerobot/agibot_world` is not public. 50fps, dual-camera,
40-dim action / 19-dim state — exercises high-DOF action validation.

| Field | Value |
|---|---|
| `repo_id` | `lerobot/unitreeh1_warehouse` |
| `codebase_version` | `v3.0` |
| `robot_type` | `unknown` |
| `fps` | 50 |
| `total_episodes` | 24 |
| `total_frames` | 11275 |
| Head download | 248.21 MB |
| Camera keys | `observation.images.cam_left`, `observation.images.cam_right` |

### Validate output (verbatim)

```
Validation: /Users/allenwu/embodied-data/data/hf_v3_samples/unitreeh1_warehouse  (format: lerobot-v3)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                  ┃ Status ┃ Detail                                                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ fps consistency        │ PASS   │ 50fps across 2 video(s), matches info.fps=50                   │
│ timestamp monotonicity │ PASS   │ 11275 rows, strict mono per episode, no spikes                 │
│ action-dim consistency │ PASS   │ 11275 rows, action dim=40, state dim=19, matches info.features │
│ frame-video alignment  │ PASS   │ 24 episode(s), 11275 rows ↔ video durations within ±1 frame    │
└────────────────────────┴────────┴────────────────────────────────────────────────────────────────┘
Result: PASS
```

Exit code: **0** (PASS).

**Interpretation**: All 24 episodes consolidated into one parquet + two mp4s;
our validator passes legitimately. Asymmetric `action=40` vs `state=19` is
intentional (action includes commanded targets the proprio doesn't see), not a
bug — confirmed by `info.features` matching.

**Robot-type oddity**: `info.robot_type = "unknown"` despite this being a
canonical Unitree H1 warehouse dataset. Not a validator concern, but a metadata
hygiene issue across the official `lerobot/*` namespace (also seen in `pusht`).

---

## Dataset 3 — `gpudad/so101_pick_cube_chunked`

**Why picked**: **community** dataset (non-`lerobot/*`), real SO-101 hardware,
massive (10990 episodes / 1.46M frames) — only one I found at 500-dataset scan
that is community-uploaded, v3, and uses real cameras. Extreme chunking
behaviour (one mp4 per episode → ~33000 video files) makes this a stress test
for the multi-file path.

| Field | Value |
|---|---|
| `repo_id` | `gpudad/so101_pick_cube_chunked` |
| `codebase_version` | `v3.0` |
| `robot_type` | `so101` |
| `fps` | 30 |
| `total_episodes` | 10990 |
| `total_frames` | 1456443 |
| `chunks_size` | **`null`** (spec says int, default 1000) |
| `data_files_size_in_mb` | **`null`** |
| `video_files_size_in_mb` | **`null`** |
| Head download | 15.11 MB (file-000 per dir; full chunk would be 879MB+) |
| Camera keys | `observation.images.{front,overhead,wrist}` |

### Validate output (verbatim)

```
Validation: /Users/allenwu/embodied-data/data/hf_v3_samples/so101_pick_cube_chunked  (format: lerobot-v3)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                  ┃ Status ┃ Detail                                                                                                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ fps consistency        │ PASS   │ 30fps across 3 video(s), matches info.fps=30                                                           │
│ timestamp monotonicity │ PASS   │ 130627 rows, strict mono per episode, no spikes                                                        │
│ action-dim consistency │ PASS   │ 130627 rows, action dim=6, state dim=6, matches info.features                                          │
│ frame-video alignment  │ FAIL   │ 32967 misalignment(s); first: ep1 observation.images.overhead: missing                                 │
│                        │        │ videos/observation.images.overhead/chunk-000/file-001.mp4                                              │
└────────────────────────┴────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘
Result: FAIL  (1 issue)
```

Exit code: **1** (FAIL).

**Interpretation (mixed)**: The FAIL is **expected given partial download** —
this dataset has one mp4 per episode per camera (10990 × 3 ≈ 33000 files) and we
fetched only `file-000` per dir. The validator correctly reports `32967`
missing files, which is exactly `(10990 − 1) × 3` (one episode of three cameras
present, the rest absent).

But the dataset itself **is** non-conformant in two real ways our validator
**did not catch** (see "Validator gaps" below):

1. `meta/info.json` has `chunks_size: null`, `data_files_size_in_mb: null`,
   `video_files_size_in_mb: null` — spec §2 declares these as `int`. The
   `DatasetInfo` dataclass would coerce defaults, but on disk these violate the
   schema and our validator never checks `info.json` field types.
2. `meta/episodes/chunk-000/file-000.parquet` has only **18 columns**, missing
   the required `tasks` column (spec §3) and the entire `stats/...` block
   (compare: `pusht` and `unitreeh1_warehouse` have 68 and 67 columns
   respectively, both including `tasks`).
3. Both `meta/tasks.parquet` and legacy `meta/tasks.json` are present — likely a
   v2.x → v3.0 migration artifact the author didn't clean up. Spec §4 says
   parquet only.

---

## Aggregate

| Status | Count | Datasets |
|---|---|---|
| PASS | 2/3 | `lerobot/pusht`, `lerobot/unitreeh1_warehouse` |
| WARN | 0/3 | — |
| FAIL | 1/3 | `gpudad/so101_pick_cube_chunked` |

### FAIL category breakdown

- **Missing-files** (1/1): `frame-video alignment` failure caused by partial
  head-only download against a one-mp4-per-episode dataset. **Not a bug in the
  dataset**; the validator was correct, but the conclusion is meaningless until
  re-run on a full mirror.

### Other (silent) findings — bugs/gaps in our validator

These are real schema violations the validator **failed to flag** and that a
v0.2 hardening pass should add:

1. **No `codebase_version` check.** `embodied-data validate` accepts any value
   for `info.codebase_version` (or its absence) and still runs the v3 checks.
   Verified by mutating `pusht` to `codebase_version: "v2.1"` — still PASSes.
2. **No `info.json` type check.** `chunks_size: null` in
   `gpudad/so101_pick_cube_chunked` is silently accepted; spec §2 says these
   are `int`. Same for `data_files_size_in_mb` / `video_files_size_in_mb`.
3. **No episode-meta `tasks` column check.** Spec §3 lists `tasks: list[str]`
   as a guaranteed column; `gpudad/so101_pick_cube_chunked` omits it entirely
   and validate still runs the alignment check successfully (it only requires
   `episode_index` and `length`).
4. **No `tasks.parquet` schema check.** All three datasets store the task
   string as `__index_level_0__` (pandas default for an unnamed index) instead
   of a column literally named `task` per spec §4. The spec wording may be
   imprecise — pandas `to_parquet` of a `Series` with a named index produces
   exactly this `__index_level_0__` layout — but if we want the documented
   schema we should either flag the missing `task` column name, or amend
   `docs/schema-lerobot-v3.md` to acknowledge the pandas-isms.
5. **Spec doc drift.** Spec §2 lists `video_files_size_in_mb` default = 200;
   `pusht` and `unitreeh1_warehouse` ship with 500. Our doc cites commit
   `cb0a944`; the value may have changed upstream since.

### What we'd test next (Sprint 3+)

- Mirror **full** `gpudad/so101_pick_cube_chunked` (~30GB) to confirm whether
  the alignment FAIL goes away once all mp4s are present, or whether there are
  genuine `from_timestamp/to_timestamp` errors hidden behind the missing-file
  path.
- Pick one dataset that uses `dtype:"image"` (PNG-embedded) instead of `"video"`
  to confirm the validator skips the alignment check sanely (none of the three
  picked use that mode).
- Try a v2.x dataset with `--format lerobot-v3` forced, to confirm we either
  reject it cleanly or migrate it correctly.
