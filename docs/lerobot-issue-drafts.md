# LeRobot ecosystem — issue drafts (pending Allen review)

All drafts below were generated against `embodied-data` rev
`960f72586d8bc68984cf514742d7d53009959d0d` on 2026-04-30. **NOT POSTED.**
Allen approves any [PUBLISH].

The Sprint 2 validation pass on three real public v3 datasets surfaced **one
candidate issue** for an upstream dataset author and **multiple v0.2 validator
tuning notes** for ourselves. There is no candidate issue against
`huggingface/lerobot` core — the upstream code is consistent with what we
observed.

---

## 1. Dataset issue — `gpudad/so101_pick_cube_chunked` ships malformed v3 metadata

**Target**: dataset-author repo
[`gpudad/so101_pick_cube_chunked`](https://huggingface.co/datasets/gpudad/so101_pick_cube_chunked)
(community-uploaded, robot_type=so101).

**Reproduction**:
```bash
# embodied-data rev: 960f72586d8bc68984cf514742d7d53009959d0d
git rev-parse HEAD  # in /Users/allenwu/embodied-data
# 960f72586d8bc68984cf514742d7d53009959d0d

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='gpudad/so101_pick_cube_chunked', repo_type='dataset',
    local_dir='./so101_full',
    allow_patterns=['meta/*'],
)"

# Then inspect:
python -c "
import json, pyarrow.parquet as pq
info = json.load(open('so101_full/meta/info.json'))
print('chunks_size', info.get('chunks_size'))
print('data_files_size_in_mb', info.get('data_files_size_in_mb'))
print('video_files_size_in_mb', info.get('video_files_size_in_mb'))
t = pq.read_table('so101_full/meta/episodes/chunk-000/file-000.parquet')
print('episode-meta cols (', len(t.schema.names), '):', t.schema.names)
print('has tasks col:', 'tasks' in t.schema.names)
"
```

**Symptom (expected output)**:
```
chunks_size None
data_files_size_in_mb None
video_files_size_in_mb None
episode-meta cols ( 18 ): ['episode_index','length','dataset_from_index','dataset_to_index','data/chunk_index','data/file_index','videos/observation.images.front/chunk_index','videos/observation.images.front/file_index','videos/observation.images.front/from_timestamp','videos/observation.images.front/to_timestamp','videos/observation.images.wrist/chunk_index','videos/observation.images.wrist/file_index','videos/observation.images.wrist/from_timestamp','videos/observation.images.wrist/to_timestamp','videos/observation.images.overhead/chunk_index','videos/observation.images.overhead/file_index','videos/observation.images.overhead/from_timestamp','videos/observation.images.overhead/to_timestamp']
has tasks col: False
```

The dataset also keeps a legacy `meta/tasks.json` (67 bytes) alongside the v3
`meta/tasks.parquet` — likely a v2.x→v3.0 migration artifact.

**Spec violations** (against `huggingface/lerobot` commit `cb0a944`):

1. `info.chunks_size`, `info.data_files_size_in_mb`, `info.video_files_size_in_mb`
   are typed `int` in `DatasetInfo` (`utils.py:108-145`). They come back as
   Python `None` when read from `info.json`, which would propagate `TypeError`
   to any consumer that arithmetics on them.
2. Per-episode metadata is missing the required `tasks: list[str]` column
   (`dataset_metadata._save_episode_metadata`, `dataset_metadata.py:404-465`).
3. Per-episode metadata also has no `stats/<feature>/{min,max,mean,std,count}`
   columns. The official `lerobot/pusht` / `lerobot/unitreeh1_warehouse` we
   compared against have 68 / 67 columns respectively (52 of which are stats
   subfields).

**Hypothesized cause**: looks like the dataset was assembled with a custom
script (or an old/forked LeRobot writer) that copies `data/` and `videos/`
correctly but skips the post-write `tasks` join and the `aggregate_stats` step.
The triple `null` on the size config keys suggests the author serialized the
`DatasetInfo` *before* the dataclass defaults were applied (e.g.,
`asdict(info)` on a hand-constructed instance with `None`s).

**Status**: pending Allen review. Suggested action when posted: open as a
discussion on the dataset card pointing to the spec sections, not a hard bug
report — the dataset is loadable for downstream training, just non-conformant.

---

## 2. v0.2 validator tuning notes (DO NOT FILE UPSTREAM — fix in our repo)

These are gaps in `embodied-data validate` itself, not LeRobot bugs. Filing
them here so they're not lost; tracker should move them into our own backlog.

### 2.1 `validate` does not enforce `codebase_version == "v3.0"`

**Reproduction**:
```bash
git rev-parse HEAD  # 960f72586d8bc68984cf514742d7d53009959d0d
cp -r data/hf_v3_samples/pusht /tmp/pusht_v25
python -c "
import json
p = '/tmp/pusht_v25/meta/info.json'
d = json.load(open(p))
d['codebase_version'] = 'v2.1'
json.dump(d, open(p,'w'))
"
embodied-data validate /tmp/pusht_v25
```

**Symptom**: returns `Result: PASS` with exit 0, despite the version having
been mutated to `v2.1`. v3 checks run regardless of declared version.

**Hypothesized cause**: `lerobot_v3.validate()`
(`src/embodied_data/validate/lerobot_v3.py:255-264`) reads `info.json` but
never checks the `codebase_version` field. Auto-detect in
`validate/__init__.py::_detect()` only looks for `meta/info.json` existence.

**Suggested fix**: add `check_codebase_version` to the v3 check list; FAIL if
absent, FAIL if not literally `"v3.0"`, with a hint to use the v2.1→v3.0
migration script.

### 2.2 `validate` does not type-check `info.json` numeric fields

**Reproduction**: `embodied-data validate data/hf_v3_samples/so101_pick_cube_chunked`
shows PASS on `fps consistency` despite `chunks_size`, `data_files_size_in_mb`,
`video_files_size_in_mb` all being `null` in `meta/info.json`.

**Symptom**: validator silently accepts; downstream consumers that try to use
those fields would crash.

**Hypothesized cause**: `_load_info()` is just `json.load`; no Pydantic model
or dataclass round-trip.

**Suggested fix**: route `meta/info.json` through a Pydantic v2 model that
mirrors the upstream `DatasetInfo` dataclass (we already depend on Pydantic);
emit one WARN per non-conforming field instead of failing hard, since spec §2
says `from_dict` is forgiving.

### 2.3 `frame-video alignment` does not check the required `tasks` column

**Reproduction**: same `so101_pick_cube_chunked` case — episode meta has no
`tasks` column, but `check_alignment` only validates that `episode_index` and
`length` are present
(`src/embodied_data/validate/lerobot_v3.py:208-214`).

**Symptom**: missing `tasks` slips through; the FAIL reported for so101 is
unrelated (it's missing mp4s due to partial download).

**Suggested fix**: add `tasks` to the required-column list in
`check_alignment`, or split off a `check_episode_meta_schema` that asserts
the spec §3 column set.

### 2.4 `tasks.parquet` schema is not validated

**Reproduction**:
```bash
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/hf_v3_samples/pusht/meta/tasks.parquet')
print(t.schema.names)  # ['task_index', '__index_level_0__']
"
```

**Symptom**: all three real datasets we tested store the task string under
`__index_level_0__` (pandas default for unnamed index) instead of a column
named `task` per `docs/schema-lerobot-v3.md` §4.

**Hypothesized cause**: probable doc-side issue. `pandas.DataFrame.to_parquet`
of a frame whose index has no `name` writes the index as `__index_level_0__`,
and the upstream LeRobot writer
(`src/lerobot/datasets/io_utils.py::write_tasks`) does not set
`tasks.index.name = "task"` before writing.

**Suggested fix (doc, not code)**: amend `docs/schema-lerobot-v3.md` §4 to
say "task string is stored as the parquet index, surfaced as
`__index_level_0__` in raw schema". If we still want to enforce, add a check
that decodes the index and confirms it contains string entries — don't check
column name.

### 2.5 Spec doc drift on `video_files_size_in_mb` default

**Observation**: `docs/schema-lerobot-v3.md` §2 lists default = `200`. Both
`lerobot/pusht` and `lerobot/unitreeh1_warehouse` ship with `500`. This is not
a validator issue, but our spec doc may be stale relative to current upstream.
Recheck `utils.py:80-82` at HEAD.

---

End of drafts. None posted.
