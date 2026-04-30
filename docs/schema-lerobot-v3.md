# LeRobotDataset v3.0 — Schema Reference

Ground truth for the embodied-data converter. All claims pinned to commit
`cb0a944` of `huggingface/lerobot` (`CODEBASE_VERSION = "v3.0"`,
[`dataset_metadata.py:53`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/dataset_metadata.py#L53)).
File references below are against this commit unless noted.

---

## 1. Directory layout

```
<root>/
├── meta/
│   ├── info.json                       # DatasetInfo dataclass, JSON-encoded
│   ├── stats.json                      # aggregated per-feature stats
│   ├── tasks.parquet                   # task_index -> task string
│   ├── subtasks.parquet                # optional
│   └── episodes/chunk-000/file-000.parquet   # per-episode metadata + per-feature stats
├── data/chunk-000/file-000.parquet     # tabular frames, multi-episode
└── videos/<camera_key>/chunk-000/file-000.mp4
```

Templates ([`utils.py:84-93`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/utils.py#L84-L93)):

```python
CHUNK_FILE_PATTERN     = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_DATA_PATH      = "data/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH     = "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_EPISODES_PATH  = "meta/episodes/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_TASKS_PATH     = "meta/tasks.parquet"
INFO_PATH              = "meta/info.json"
STATS_PATH             = "meta/stats.json"
```

### Chunking

A "chunk" holds up to `chunks_size` files; index rolls over via
`update_chunk_file_indices` ([`utils.py:233-238`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/utils.py#L233-L238)).
A new file is opened when the current file would exceed
`data_files_size_in_mb` (parquet) or `video_files_size_in_mb` (video).
**Episodes are not split across files** — each episode lives entirely in one
parquet and one mp4 per camera.

| Constant | Default | Source |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `1000` | `utils.py:80` |
| `DEFAULT_DATA_FILE_SIZE_IN_MB` | `100` | `utils.py:81` |
| `DEFAULT_VIDEO_FILE_SIZE_IN_MB` | `200` | `utils.py:82` |

---

## 2. `meta/info.json`

Backed by `DatasetInfo` dataclass
([`utils.py:108-145`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/utils.py#L108-L145)).
`from_dict` ignores unknown keys for forward-compat.

| Field | Type | Default | Notes |
|---|---|---|---|
| `codebase_version` | str | — | MUST be `"v3.0"` |
| `fps` | int | — | must be > 0 |
| `features` | dict[str, dict] | — | feature_key → `{dtype, shape, names}` (+ optional `info`) |
| `total_episodes` | int | 0 | |
| `total_frames` | int | 0 | |
| `total_tasks` | int | 0 | |
| `chunks_size` | int | 1000 | |
| `data_files_size_in_mb` | int | 100 | |
| `video_files_size_in_mb` | int | 200 | |
| `data_path` | str | `DEFAULT_DATA_PATH` | template |
| `video_path` | str \| null | `DEFAULT_VIDEO_PATH` | `null` if no videos |
| `robot_type` | str \| null | None | |
| `splits` | dict[str, str] | `{}` | writer sets `{"train": "0:N"}` |

> **Uncertain**: the spec brief lists `total_videos` and `total_chunks`. These
> exist in v2.x `info.json` but are **not** fields on the v3 `DatasetInfo`
> dataclass — `from_dict` accepts and silently drops them. Do not emit.

### `features` entry shape

```json
"observation.state": {"dtype":"float32","shape":[14],"names":["joint_0",...]},
"observation.images.top": {"dtype":"video","shape":[3,480,640],
  "names":["channel","height","width"],
  "info":{"video.fps":30,"video.codec":"av1","video.pix_fmt":"yuv420p",
          "video.height":480,"video.width":640,"video.channels":3,
          "video.is_depth_map":false}},
"action": {"dtype":"float32","shape":[14],"names":[...]}
```

`info` for video features is auto-populated on first encode by
`update_video_info()` calling `get_video_info()`
([`video_utils.py:989-1019`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/video_utils.py#L989)).

### Auto-populated default features

Added by the recording pipeline; converter must inject these
([`utils/constants.py: DEFAULT_FEATURES`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/utils/constants.py)):

```python
DEFAULT_FEATURES = {
    "timestamp":     {"dtype":"float32","shape":(1,),"names":None},
    "frame_index":   {"dtype":"int64",  "shape":(1,),"names":None},
    "episode_index": {"dtype":"int64",  "shape":(1,),"names":None},
    "index":         {"dtype":"int64",  "shape":(1,),"names":None},
    "task_index":    {"dtype":"int64",  "shape":(1,),"names":None},
}
```

---

## 3. `meta/episodes/chunk-XXX/file-XXX.parquet`

One row per episode. Columns are dynamic; the following are guaranteed
(authority: `_save_episode_metadata`
[`dataset_metadata.py:404-465`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/dataset_metadata.py)
+ migration-script comment block at
[`scripts/convert_dataset_v21_to_v30.py:96-130`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/convert_dataset_v21_to_v30.py)):

| Column | Type | Semantics |
|---|---|---|
| `episode_index` | int64 | zero-based episode id |
| `tasks` | list[str] | unique task strings used in this episode |
| `length` | int64 | number of frames |
| `meta/episodes/chunk_index` | int64 | self-locator |
| `meta/episodes/file_index` | int64 | self-locator |
| `data/chunk_index` | int64 | which `data/chunk-XXX/` holds frames |
| `data/file_index` | int64 | which `file-XXX.parquet` |
| `dataset_from_index` | int64 | global frame index of episode start |
| `dataset_to_index` | int64 | global frame index of episode end (exclusive) |
| `videos/<vid_key>/chunk_index` | int64 | per video feature |
| `videos/<vid_key>/file_index` | int64 | per video feature |
| `videos/<vid_key>/from_timestamp` | float64 | episode's start offset within multi-episode mp4 (s) |
| `videos/<vid_key>/to_timestamp` | float64 | episode's end offset (s) |
| `stats/<feature_key>/{min,max,mean,std,count}` | nested | per-episode stats; stripped on `load_episodes` |

**Video frame lookup.** Parquet rows do **not** store a video path or row id.
The reader resolves `(ep_idx, vid_key) → mp4 path` via the episode metadata
(`videos/<vid_key>/{chunk,file}_index`), then seeks by
`from_timestamp + frame_offset_seconds` inside that mp4
([`dataset_reader.py:233-242`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/dataset_reader.py#L233-L242)).
Per-frame `timestamp` in `data/*.parquet` = `frame_index / fps`; per-episode
`from_timestamp` = cumulative duration of preceding episodes in the same mp4.

---

## 4. `meta/tasks.parquet`

**Not jsonl in v3.** Pandas DataFrame written via `tasks.to_parquet(...)`
([`io_utils.py: write_tasks`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/io_utils.py)):

| Column / index | Type | Notes |
|---|---|---|
| index `task` | str | natural-language description (DataFrame index) |
| `task_index` | int64 | sequential id, 0-based |

Legacy v2.x `tasks.jsonl` (`{"task_index": int, "task": str}`) is read only by
the migration script.

---

## 5. `data/chunk-XXX/file-XXX.parquet`

Multi-episode frame table. Schema is `get_hf_features_from_features(features)`
plus default features
([`feature_utils.py:31-67`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/feature_utils.py#L31-L67)).

### Required columns

| Column | dtype | Shape | Source |
|---|---|---|---|
| `timestamp` | float32 | (1,) | `frame_index / fps` |
| `frame_index` | int64 | (1,) | 0-based within episode |
| `episode_index` | int64 | (1,) | matches metadata |
| `index` | int64 | (1,) | global, monotonically increasing |
| `task_index` | int64 | (1,) | resolved via `tasks.parquet` |
| `observation.state` | float32 | (D,) | length declared in `features` |
| `action` | float32 | (D,) | |

Video features (`dtype:"video"`) are **excluded from parquet**; image features
(`dtype:"image"`) are stored as embedded PNG bytes via `datasets.Image()`.

### LeRobot dtype → HF Datasets mapping

| LeRobot dtype | shape | HF type |
|---|---|---|
| numpy dtype string | `(1,)` | `Value(dtype)` |
| numpy dtype string | `(N,)` | `Sequence(length=N, feature=Value(dtype))` |
| numpy dtype string | 2D–5D | `Array2D` … `Array5D` |
| `"image"` | any | `Image()` (PNG bytes embedded) |
| `"video"` | any | excluded from parquet |
| `"string"` | — | string |

---

## 6. Video files

Encoded by `encode_video_frames`
([`video_utils.py`](https://github.com/huggingface/lerobot/blob/cb0a944941ea4a64de120bb36b2b8bc2b6953520/src/lerobot/datasets/video_utils.py)):

| Setting | Default | Notes |
|---|---|---|
| Container | MP4 | via PyAV |
| `vcodec` | `"libsvtav1"` | also: `h264`, `hevc`, `auto`, hw encoders (`h264_videotoolbox`, `*_nvenc`, `h264_vaapi`, `h264_qsv`) |
| `pix_fmt` | `"yuv420p"` | forced when `libsvtav1`/`hevc` paired with `yuv444p` |
| GOP `g` | `2` | small → seek-friendly |
| `crf` | `30` | mapped per codec |
| `preset` (libsvtav1) | `"12"` | |
| FPS | `info.fps` | passed as ffmpeg stream rate |

Sequential episodes per camera are concatenated by `concatenate_video_files`
until `video_files_size_in_mb` is exceeded. Episode boundaries inside an mp4
are recovered via `videos/<vid_key>/from_timestamp`/`to_timestamp` columns.

### fps enforcement & non-monotonic timestamps

- `fps` is required positive int, validated in `DatasetInfo.__post_init__`.
- Per-frame `timestamp = frame_index / fps`. Per-episode `from_timestamp` =
  previous episode's `to_timestamp`.
- Failure mode: when concatenated mp4 fragments have non-monotonic DTS, the
  PyAV muxer raises `av.error.ValueError: [Errno 22] Invalid argument` —
  see [#2158](https://github.com/huggingface/lerobot/issues/2158)
  (4th-episode concat fails) and [#2689](https://github.com/huggingface/lerobot/issues/2689)
  (post-conversion training shows "spark of joint actions"; root cause is
  drifting timestamps).
- Decoder enforces tolerance on read: `decode_video_frames` raises
  `FrameTimestampError` if the closest decoded frame is outside `tolerance_s`
  of the queried timestamp.

---

## 7. `meta/stats.json`

Aggregated stats (post-`aggregate_stats`), serialized via `serialize_dict`
(numpy → nested lists).

```json
{
  "observation.state": {
    "min":[...],"max":[...],"mean":[...],"std":[...],"count":[N]
  },
  "observation.images.top": {
    "min":[[[r]],[[g]],[[b]]],   // shape (C,1,1), normalized to [0,1]
    "max":[...],"mean":[...],"std":[...],"count":[N]
  }
}
```

Image stats use channel-first `(C,1,1)` arrays, pixel-normalized (÷255).
`count` is always a 1-element list. Per-episode stats also live inside
`meta/episodes/.../file-XXX.parquet` under `stats/<feature>/...` columns;
`stats.json` is the running aggregate.

> **Uncertain**: the v2.1→v3.0 migration script comment says "Remove the
> deprecated `stats.json`", yet `dataset_metadata.save_episode` calls
> `write_stats` on every save, and v3 datasets in the wild contain
> `stats.json`. Treat as required to emit, non-fatal if missing on read
> (`load_stats` returns `None`).

---

## 8. Differences from v2.0 / v2.1 (breaking)

Source: comment block at
[`scripts/convert_dataset_v21_to_v30.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/convert_dataset_v21_to_v30.py).

| Aspect | v2.0 / v2.1 | v3.0 |
|---|---|---|
| Data file naming | `data/chunk-000/episode_000000.parquet` (1 episode/file) | `data/chunk-000/file-000.parquet` (multi-episode) |
| Video file naming | `videos/chunk-000/<CAMERA>/episode_000000.mp4` | `videos/<CAMERA>/chunk-000/file-000.mp4` (chunk under camera) |
| Episode metadata | `meta/episodes.jsonl` with `{episode_index, tasks, length}` | `meta/episodes/chunk-000/file-000.parquet` (rich, see §3) |
| Episode stats | `meta/episodes_stats.jsonl` | merged into `meta/episodes/.../file-XXX.parquet` |
| Tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| `info.json` extras | had `total_videos`, `total_chunks` | dropped from dataclass |
| `stats.json` | present in v2.0; migration comment "remove deprecated" | still emitted by current v3 writer (see §7) |
| Migration | `convert_dataset_v1_to_v2.py` | `convert_dataset_v21_to_v30.py`; **no v2.0→v3.0 script** ([#2446](https://github.com/huggingface/lerobot/issues/2446)) — must chain v2.0→v2.1→v3.0 |
| Compatibility | — | reading v2.1 raises `BackwardCompatibilityError` |

---

## 9. Known footguns

| # | Symptom | Issue | Mitigation |
|---|---|---|---|
| F1 | `av.error.ValueError: [Errno 22]` during video concat (non-monotonic DTS) | [#2689](https://github.com/huggingface/lerobot/issues/2689), [#2158](https://github.com/huggingface/lerobot/issues/2158) | Re-encode each episode mp4 from scratch with `g=2`; never reuse upstream timestamps. Validate `from_timestamp` strictly increases per camera per file. |
| F2 | `OSError: [Errno 39] Directory not empty` on the 2nd `save_episode` | [#2158](https://github.com/huggingface/lerobot/issues/2158) | Don't reuse HF dataset cache dirs across episodes; each temp encode under its own `tempfile.mkdtemp()` and `rmtree` only that path. |
| F3 | No v2.0 → v3.0 path | [#2446](https://github.com/huggingface/lerobot/issues/2446) | Either ingest v2.0 directly or chain v2.0→v2.1 first. Document accepted v2.x minor versions explicitly. |
| F4 | `normalize_*.buffer_*.{mean,std}` shape mismatch on policy load | [#1329](https://github.com/huggingface/lerobot/issues/1329) | Validate `len(stats[key]["mean"]) == features[key]["shape"][0]` for `observation.state` and `action`. |
| F5 | `fps` stripped from scalar default features after merge / episode delete | [#2679](https://github.com/huggingface/lerobot/issues/2679) | Always read top-level `info.fps` as authoritative; don't rely on `features[k].get("fps")`. |

---

## 10. Sources (canonical)

All inline links target commit
[`cb0a944`](https://github.com/huggingface/lerobot/tree/cb0a944941ea4a64de120bb36b2b8bc2b6953520)
(PR #3472, typed `DatasetInfo` refactor). Migration script reference:
[`convert_dataset_v21_to_v30.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/convert_dataset_v21_to_v30.py).
HF blog: <https://huggingface.co/blog/lerobot-datasets-v3>. Issues cited in §9.
