# Changelog

All notable changes to **embodied-data** are documented in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-04-30

Patch release. v0.1.0 was implicitly validated only against the open
`agibot-world/AgiBotDigitalWorld` simulator. v0.1.1 hardens v0.1.0's
AgiBot path against real-hardware **AgiBot Beta** data discovered during
post-release fixture acquisition: filename typo (`proprio_stats.h5` vs
`proprio_states.h5`), 14-dim joint shape (vs sim's 34-dim), int64 ns
timestamps (vs sim's float32 sec), and a list-of-episodes `task_info_*.json`
layout (vs sim's single-episode dict). v0.1.1 ships a refuse-and-document
guard for forward conversion plus fixes for several silent miscompiles in
`preview` / `validate` / `inspect`. Real-Beta forward conversion is v0.2
work — see PR #1 for the in-progress design.

### Fixed

- **Filename glob** — accept both `proprio_states.h5` (DigitalWorld) and `proprio_stats.h5` (Beta) across `convert`, `validate`, `preview`. Closes silent "format unknown" failure on real Beta data.
- **Detect-and-refuse real-robot data** — when AgiBot input has 14-joint shape or missing `state/joint.attrs["name"]`, `convert` exits with a clear "v0.1 supports DigitalWorld sim only; v0.2 roadmap covers Beta/Alpha" message instead of a raw traceback.
- **`task_info` list-of-episodes** — `convert` and `preview` correctly handle Beta's list-of-399-episode-dicts JSON layout. Falls back to first entry's `task_name`. No more silent "(unknown)" task.
- **`preview` reports actual joint count** — no longer claims "State dim: 22 (raw 14)" when the schema can't subselect. Surfaces the mismatch.
- **`preview` reads `robot_type` from h5 attrs** — no more hardcoded "a2d"; falls back to "unknown" when attrs missing.
- **`convert` error path catches `KeyError` / `ValueError`** — no more raw Rich tracebacks on schema mismatches.
- **`inspect` prints `.attrs`** — group/dataset attrs now appear in the dump (cheap diagnostic for catching missing-`name`-attr issues).

### Known limitations (still v0.1.x)

- AgiBot Beta/Alpha real-robot **forward conversion** is not supported. Refuse-and-document is the v0.1.1 behavior; full ingest is v0.2 (see [PR #1](https://github.com/allenwu-blip/embodied-data/pull/1)).
- `validate` still treats `int64` ns timestamps as a silent PASS on monotonicity (rate inference would lie). Fixing the WARN message is v0.2.

## [0.1.0] — 2026-04-30

First public release. Bidirectional converter and validator for **AgiBot World ↔
LeRobot v3** datasets.

### Added

- **`convert` command**
  - `agibot → lerobot-v3`, single episode: 22-dim joint subselect (head + body + dual arms + effectors), first-difference action, recomputed `timestamp = frame_index / 30`, head camera re-encoded to h264 with `bf=0 g=2` to dodge LeRobot v3 footgun F1 (mp4 muxer DTS reorder).
  - `agibot → lerobot-v3`, **multi-episode batch** with `--max-episodes`, `--resume` (UUID-keyed, idempotent on rerun), `--workers` (`ProcessPoolExecutor` for h5 reads; video encode stays main-process to avoid PyAV pickle + chunk-dir filename races). OOM-safe streaming, `rich.Progress` UI.
  - `lerobot-v3 → agibot`, **reverse converter**, single episode. Zero-fills the 12 dropped passive joints; round-trip preserves the 22 forwarded joints within `rtol=1e-5 atol=1e-6` and timestamps exactly.
- **`validate` command** — five checks against a LeRobot v3 directory:
  1. **schema conformance** — `codebase_version == "v3.0"`, non-null `chunks_size` / `data_files_size_in_mb` / `video_files_size_in_mb`, required `tasks` column in episode metadata, `tasks.parquet` column-or-index format.
  2. **fps consistency** — `info.fps` vs. every video stream's `average_rate`.
  3. **timestamp monotonicity** — strict per-episode monotone, surfaces LeRobot [#2689](https://github.com/huggingface/lerobot/issues/2689) spike footgun.
  4. **action-dim consistency** — parquet `action` and `observation.state` widths match `info.features`.
  5. **frame-video alignment** — episode `length / fps` vs. mp4 duration within ±1 frame (catches AgiBot-World [#149](https://github.com/OpenDriveLab/AgiBot-World/issues/149)).
  AgiBot mode also raises a `WARN` on the upstream 60Hz `/timestamp` bug.
- **`preview` command** — Rich-formatted stats table (frames, fps, state/action dims, camera list, task names) for either format.
- **`inspect` command** — schema dump for a single HDF5 (groups + datasets + dtypes + attrs) or parquet (columns + row count + first 3 rows) file. Dev tool.
- Top-level **`--json`** flag — machine-readable JSON on stdout for any command.
- Top-level **`--version`** flag — prints version, git short SHA, and build date.
- Actionable error suggestions on every failure path ("expected `meta/info.json` (lerobot-v3) or `proprio_states.h5` (agibot) in path", etc.).
- **Schema reference docs** authored from `huggingface/lerobot @ cb0a944` and the AgiBot `convert_to_lerobot.py` source: [`docs/schema-lerobot-v3.md`](docs/schema-lerobot-v3.md), [`docs/schema-agibot.md`](docs/schema-agibot.md), [`docs/schema-mapping.md`](docs/schema-mapping.md).
- **HF dataset survey** ([`docs/hf-dataset-findings.md`](docs/hf-dataset-findings.md)) — three real public v3 datasets validated end-to-end (2 PASS, 1 FAIL with reproducible cause).
- **Upstream issue triage** ([`docs/issues-coverage.md`](docs/issues-coverage.md)) — 10 issues triaged; 5 in-scope addressed by this release.

### Coverage

In-scope upstream issues addressed by v0.1.0:

| Issue | Title | Resolution |
| --- | --- | --- |
| [`OpenDriveLab/AgiBot-World#18`](https://github.com/OpenDriveLab/AgiBot-World/issues/18) | convert script ValueError + visualizer TypeError | `convert` + `validate` |
| [`OpenDriveLab/AgiBot-World#124`](https://github.com/OpenDriveLab/AgiBot-World/issues/124) | `KeyError: Column 'actions'` (v2.1) | `convert` (flat action column) |
| [`OpenDriveLab/AgiBot-World#149`](https://github.com/OpenDriveLab/AgiBot-World/issues/149) | frame/video misalignment in 9 tasks | `convert` + `validate` (frame-video alignment check) |
| [`huggingface/lerobot#2158`](https://github.com/huggingface/lerobot/issues/2158) | local v3 conversion (cache + invalid-timestamp) | `convert` (single-shot v3 write, monotonic PTS) + `validate` (timestamp monotonicity check) |
| [`huggingface/lerobot#2689`](https://github.com/huggingface/lerobot/issues/2689) | ALOHA sim v2.1→v3.0 spark joint actions (partial) | `validate` flags action-dim drift / timestamp non-monotonicity on the converted dataset; ALOHA HDF5 ingest itself is out of scope |

5 explicitly out-of-scope issues remain documented and politely declined in [`docs/issue-comments-drafts.md`](docs/issue-comments-drafts.md).

### Known limitations

- **AgiBot side, head camera only.** The 7 fisheye/hand cameras and the depth stream (uint16 mm PNG) are documented but not yet implemented.
- **Reverse converter is single-episode.** Multi-episode reverse is v0.2.
- **One-episode-per-parquet** in batch mode. True size-based multi-episode-per-parquet rollover is v0.2.
- **Frame/video alignment uses an episode-metadata proxy** (length/fps × duration). Per-frame PTS drift inside a video file would not be caught by v0.1.

### Verified on

- Python 3.12, macOS arm64
- 50 tests passing
- Real data: paired episode from open `agibot-world/AgiBotDigitalWorld` (375 frames, 1 task, 1 head camera) + 3 HF community v3 datasets validated end-to-end (2 PASS, 1 FAIL with reproducible cause documented in `docs/hf-dataset-findings.md`).

### Acknowledgments

Built on top of the schemas published by the HuggingFace LeRobot team and the
OpenDriveLab AgiBot team.

[0.1.1]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.1
[0.1.0]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.0
