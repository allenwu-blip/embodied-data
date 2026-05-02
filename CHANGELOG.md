# Changelog

All notable changes to **embodied-data** are documented in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [0.3.1] — 2026-05-01

Quality-of-life patch release. Three new CLI flags, one Beta task-name
resolution fix, and OSS-grade repo hygiene (issue templates, nightly CI,
Discussions, CONTRIBUTING / CoC / SECURITY). **Datasets produced by v0.3.0
work unchanged in v0.3.1** — this release is purely additive.

### Added

- `convert --dry-run` — print a conversion plan (format pair, detected variant, single vs batch mode, episode count, estimated total frames, estimated output size in MB) without writing any files. Hard-fails with exit 2 if the destination would have been overwritten by a real run, so dry-run can't be used to silently preview a destructive overwrite. Reverse pair (`lerobot-v3 → agibot`) is not supported in dry-run yet — exits 2 with a scoped error.
- `convert --verify` — automatically run `validate` on the destination after a successful convert and propagate the result to the convert exit code. Skipped when `--dry-run` is also passed (nothing to validate) and when the format pair is reverse (no output-side validator yet).
- `inspect <dataset_dir> --summary` — high-level overview of a LeRobot v3 dataset: robot type, fps, episode / frame / duration totals, state and action dims, per-camera resolution + codec, disk size, and a mini run of all five `validate` checks with PASS/FAIL/SKIP status. Exits non-zero on overall FAIL. `--json --summary` emits the full summary as a JSON dict for programmatic consumption.

### Changed

- `_resolve_beta_task_name` now walks up to four ancestors (was only `src` and `src.parent`). Single-episode CLI invocations like `embodied-data convert <root>/<task>/<ep_id>/ ...` now find the canonical `<root>/task_info_<task>.json` and produce a `meta/tasks.parquet` with the real task name instead of `"unknown"`. Resolves the v0.2.x patch backlog item.

### Fixed

- `validate` no longer false-FAILs on `lerobot/pusht` and other multi-episode-per-video LeRobot v3 datasets. v0.3.0's frame-count cross-check incorrectly compared whole-video frame count to single-episode length, producing one FAIL per episode on the official quick-start dataset (`pusht`'s 25650-frame video shared across 206 episodes). v0.3.1 scopes the frame-count check to one-episode-per-mp4 datasets only (detected via `from_timestamp ≈ 0` AND `to_timestamp ≈ frames/fps`). Multi-episode-per-mp4 datasets fall back to the duration check, which correctly validates per-episode slices via `(to_ts - from_ts) ≈ length/fps`.

### Repo hygiene (no user-visible behavior change)

- GitHub Discussions enabled with 4 sticky welcome threads (Announcements / Q&A / Show and tell / Ideas).
- Issue templates (`bug_report.yml`, `feature_request.yml`, `question.yml`) and a PR template added under `.github/`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md` added at repo root.
- Nightly CI workflow (`nightly.yml`) — cron 04:00 UTC across Python 3.11 / 3.12 / 3.13 with auto-issue creation on cron-triggered failure.
- README badges expanded to 5 (PyPI version + monthly downloads + CI + Python pyversions + License) and a new `convert-output.svg` screenshot embedded under "What it does".

## [0.3.0] — 2026-05-01

Minor release. Closes the largest user-facing gap from v0.2.0's Known
Limitations: Beta / Alpha LeRobot v3 datasets emitted by `embodied-data
convert` now contain re-encoded `observation.images.head_color` video
alongside proprio. **v0.3.0 datasets are usable for VLA fine-tuning
end-to-end** — videos are no longer `null`.

### Added

- **`observation.images.head_color` for Beta / Alpha.** `convert_agibot_beta_to_lerobot_v3`
  (single-episode) and `convert_agibot_beta_batch` (multi-episode) now
  re-encode `<src>/videos/head_color.mp4` through the LeRobot v3 video
  contract — h264, `bf=0`, `g=2`, `yuv420p`, monotonic PTS. Output lives
  at `videos/observation.images.head_color/chunk-{i:03d}/file-{j:03d}.mp4`,
  `info.json` declares the video feature, and per-episode video columns
  (`videos/<key>/chunk_index`, `file_index`, `from_timestamp`,
  `to_timestamp`) are emitted alongside proprio. End-to-end:
  `embodied-data convert <beta_episode_dir> <dst> --from agibot --to lerobot-v3`
  now produces a v3 dataset directly consumable by VLA fine-tuning
  pipelines.
- **Shared `_video.py` helpers.** `reencode_video` and `probe_video` factor
  out the v0.1 sim re-encode path so both pipelines share a single
  enforcement of the LeRobot v3 video constraints.
- **Hard-fail on `validate` for declared-but-missing video.** When
  `info.features` declares `dtype: video`, `frame-video alignment`
  now FAILs (was SKIP) on missing mp4 files, missing episode-meta video
  columns, codec decode errors, or frame-count divergence >1 frame.
  Proprio-only datasets (no `dtype: video` in features) still SKIP
  cleanly.

### Changed

- **Beta batch all-or-nothing video.** If any pending or already-committed
  episode in a batch has `videos/head_color.mp4`, the dataset declares
  the head_color feature; episodes lacking the upstream mp4 are logged
  to `.beta_batch_errors.jsonl` and skipped (preserving v0.2's
  per-episode error model). If no episode has video, output is
  proprio-only (legacy v0.2 behavior unchanged).
- **`_commit_beta_episode` re-encodes video before writing data.** A
  missing/broken upstream mp4 fails the commit before any state-laden
  parquets land, keeping `data/`, `meta/episodes/`, and `uuid_map.parquet`
  in sync even on partial-failure batches.

### Known limitations (still v0.3.x)

- **Multi-camera support** (`fisheye`, `hand_left/right`, `back_left/right`).
  v0.3.0 ships head_color only — other cameras land in v0.3.1.
- **Sparse `action/*/index` companions** still dropped silently — v0.3.2
  candidate.
- **`state/end/*` end-pose flattening** (32-dim) — v0.3.3 candidate.
- **Reverse `lerobot-v3 → agibot-beta`** still not implemented.

## [0.2.0] — 2026-04-30

Minor release. Adds first-class support for real-hardware AgiBot captures
(`agibot-world/AgiBotWorld-Beta`, also `agibot-world/AgiBotWorld-Alpha` per
empirical schema equivalence) on top of v0.1.x's sim-only DigitalWorld
pipeline. CLI surface unchanged — the `convert` command auto-detects the
variant via `detect_agibot_variant` and routes to the right converter.

### Added

- **Beta single-episode forward conversion** — `convert_agibot_beta_to_lerobot_v3`
  reads 14 joints + 2-dim effector + 2-dim head + 2-dim waist into a 20-dim
  `observation.state`, computes first-difference action, recomputes
  `frame_index/30` timestamps (Beta's `int64` ns Unix-epoch column is
  intentionally discarded per LeRobot v3 invariant), and resolves task name
  from Beta's `task_info_<task>.json` list-of-episodes layout. No videos
  in this first cut (the Beta sample we hold ships proprio + metadata only).
- **Beta multi-episode batch** — `convert_agibot_beta_batch` mirrors the
  sim batch's structure: `--max-episodes`, `--resume` (UUID-keyed,
  idempotent via `meta/extra/uuid_map.parquet`), `--workers`
  (`ProcessPoolExecutor` on h5 reads, single-process commit). Failed
  episodes log to `<dst>/.beta_batch_errors.jsonl` and the run continues.
- **Schema-detect dispatcher** — `embodied_data._agibot_paths.detect_agibot_variant`
  returns `digitalworld | beta | alpha | unknown`. The `convert` command
  uses this to auto-route: sim DigitalWorld → existing pipeline (unchanged);
  Beta single-episode dir or task root → Beta path (auto-batches when src
  is a task root or batch flags are set); Alpha-named path → Beta path with
  a one-line console note. Unknown layouts emit a structured error with a
  `schema_summary` of what was actually found.
- **Schema reference docs reorged per variant** — `docs/schema/`:
  `overview.md` (variant detection rules + Alpha ≡ Beta empirical finding +
  per-variant coverage matrix), `digitalworld.md` (sim variant — content
  from the former monolithic `schema-agibot.md`), `beta.md` (Beta + Alpha
  reference). The pre-v0.2 `docs/schema-agibot.md` is kept as a stub
  redirect for legacy references.

### Changed

- **Alpha is no longer refused.** v0.1.1's "v0.2 follow-up milestone" stub
  error is replaced with auto-routing through the Beta converter.
  Equivalence verified empirically on 2026-04-30 (Alpha task 389/episode
  656913 vs Beta task 675/episode 936938 head-to-head h5 diff): identical
  joint shape `(N, 14)`, identical missing `state/joint.attrs["name"]`,
  identical `int64` ns timestamps, identical state subgroup set.

### Known limitations (deferred to v0.3+)

- **Videos** for Beta/Alpha. v0.2 emits `video_path: null`. Video ingest
  needs separate fixture acquisition (per-episode tars upstream are several
  GB).
- **Sparse `action/*/index` companions** (Beta has these for `joint`,
  `effector`, `end`, `robot`, `head`, `waist`). v0.2 drops them silently;
  v0.3 will surface as `auxiliary.*.mask` features.
- **`state/end/*` end-pose flattening** into `observation.state.end_pose`
  (32-dim) — v0.2.x candidate if user demand surfaces.
- **Reverse `lerobot-v3 → agibot-beta`** — single-episode reverse for sim
  is in v0.1; Beta-flavoured reverse is v0.3.
- **`--joint-names <file.json>` override** for users whose Beta task has
  different joint ordering than `JOINT_14_BETA = [arm_l_j1..7, arm_r_j1..7]`
  (v0.2's best-guess constant — see `docs/schema/beta.md` §7).
- **Per-frame raw timestamp preservation** under `auxiliary.timestamp_raw`
  (v0.2 discards Beta's `int64` ns column in favour of LeRobot v3's
  `frame_index/fps` invariant).

### Test count

- Sprint 3 closeout (v0.1.1 GA): 64 passed
- Post-M1 (Beta single-episode): 71 passed
- Post-M2 (dispatcher): 86 passed + 1 skipped
- Post-M3 (Beta batch): 96 passed + 1 skipped
- Post-Alpha verification + schema reorg + design §5 tests: **98 passed + 1 skipped**

### Acknowledgments

Built on top of the schemas published by the HuggingFace LeRobot team and
the OpenDriveLab AgiBot team. Empirical Alpha access made the
"schemas equivalent per upstream README" claim verifiable rather than
assumed.

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

[0.3.1]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.3.1
[0.3.0]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.3.0
[0.2.0]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.2.0
[0.1.1]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.1
[0.1.0]: https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.0
