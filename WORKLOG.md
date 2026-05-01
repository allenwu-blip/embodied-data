# WORKLOG

## Sprint 1 — 2026-04-30

### Done
- v0.0.1 scaffold landed: typer CLI, MIT LICENSE, GitHub Actions (uv + ruff + pytest), pre-commit hooks. Repo public at https://github.com/allenwu-blip/embodied-data.
- Triaged 10 upstream issues into `docs/issues-coverage.md` (5 in-scope for v0.1, 5 explicitly out-of-scope).
- Authored ground-truth schema docs:
  - `docs/schema-lerobot-v3.md` — LeRobotDataset v3 directory layout, `info.json`, parquet/video schema, 9 footguns.
  - `docs/schema-agibot.md` — AgiBot DigitalWorld HDF5 tree, joint subselection (34→22), 8-camera layout, 60Hz timestamp upstream bug, 8 bugs in upstream `convert_to_lerobot.py`.
  - `docs/schema-mapping.md` — explicit field-by-field mapping with provenance citations.
- Pivoted sample dataset from gated `AgiBotWorld-Beta` to open `AgiBotDigitalWorld`; one paired episode (`proprio_states.h5` + 8 mp4s + depth, 124MB total) at `data/agibot_sample/`.
- Implemented v0.0.2 minimal-path converter `agibot → lerobot-v3`: 22-dim state subselect, first-difference action, `frame_index/30` timestamps, single head camera re-encoded to h264 (B-frames disabled to dodge v3 footgun F1).
- Implemented `validate` with 4 checks (fps consistency, timestamp monotonicity, action-dim, frame-video alignment) and a `WARN` for the AgiBot 60Hz timestamp bug.
- Implemented `preview` with rich Table output for both formats.
- 23 tests green; ruff clean; CI green on every push.

### End-state verification (one-shot E2E)
```
$ uv run embodied-data convert data/agibot_sample/.../000aa0b4-... /tmp/out --from agibot --to lerobot-v3
done: 375 frames, 1 task, 1 camera (observation.images.top_head) → /tmp/out
$ uv run embodied-data validate /tmp/out
Result: PASS
$ uv run embodied-data preview /tmp/out
Format: lerobot-v3 | Total frames: 375 | fps: 30 | State dim: 22 | Action dim: 22
```

### Blockers / open items
- HF_TOKEN [CRED] still pending — gated `AgiBotWorld-Beta` not yet accessible. Non-blocking for v0.1 (DigitalWorld is structurally equivalent), nice-to-have for canonical fixture in v0.1.1.
- Subagent D timed out mid-implementation; converter was finished by Tech Lead directly. No regression — handoff was clean because schema docs were already authoritative.
- E's `frame-video alignment` check on lerobot-v3 uses an episode-metadata proxy (length/fps × duration) instead of decoding every frame. Catches the AgiBot-World#149 row/frame-count footgun; would not catch a per-frame timestamp drift inside a video. Acceptable for v0.1.

### Next sprint candidates (do NOT pre-batch — pick at next sprint kickoff)
- Multi-camera support (7 fisheye + hand cameras)
- Depth as `observation.images.cam_top_depth` (uint16 mm PNG-bytes)
- Multi-episode batching with chunk rollover
- Bidirectional: `lerobot-v3 → agibot` (lower priority — fewer downstream users)
- Drafting comments to post on the 5 in-scope upstream issues, awaiting Allen's [PUBLISH] approval

---

## Sprint 2 — 2026-04-30 (v0.1 GA scope locked, plumbing ready)

### Done

**P0 (the v0.1 trust moat)**
- **Batch episode pipeline** — `convert_agibot_batch` with `--max-episodes / --resume / --workers`, OOM-safe streaming, rich.Progress, idempotent on rerun (uuid map under `meta/extra/uuid_map.parquet`). Subagent A. Stats recomputed from on-disk parquets at finalize to dodge float32 path divergence.
- **Reverse converter `lerobot-v3 → agibot`** — Subagent B stalled at 600s with no real output (second timeout this project; same outcome as Sprint 1 Subagent D). Tech Lead implemented directly: `convert_lerobot_v3_to_agibot` zero-fills the 12 dropped passive joints, refuses multi-episode v3 cleanly, byte-copies the v3 mp4 to the AgiBot sibling layout. **Round-trip test** (`tests/test_round_trip.py`): AgiBot → LeRobot v3 → AgiBot preserves all 22 forwarded joints within `rtol=1e-5 atol=1e-6` and timestamps exactly.
- **Real HF dataset validation** — Subagent C surveyed 3 public v3 datasets:
  - `lerobot/pusht` → PASS (5/5 checks after schema check addition)
  - `lerobot/unitreeh1_warehouse` → PASS
  - `gpudad/so101_pick_cube_chunked` → FAIL (alignment, expected — head-only download against one-mp4-per-episode dataset; FAIL also fires on missing `tasks` column and null int fields, all real).
  - 5 silent gaps in our validator surfaced; **3 patched in this sprint** via the new `check_schema_conformance` (codebase_version, info.json int types, episode-meta `tasks` column, tasks.parquet column flexibility), 2 deferred to v0.2 (full mp4 corpus, image-PNG mode coverage).

**P1**
- **CLI polish** — Subagent E: top-level `--json`, `--version` with git short-hash + build date, new `inspect` command for h5/parquet schema dump, error suggestions across all commands ("AgiBot expects parallel meta_info/<task>/<uuid>/ and observations/...", "v3 dataset must have at least the top_head camera for v0.1 reverse", etc.).
- **PyPI build plumbing** — Subagent D: `uv build` produces wheel + sdist; `twine check` PASSES on both; `scripts/check_version.py` + CI step + `tags: ['v*']` push trigger guard against version-tag drift; `docs/release-checklist.md` (7-step) and `docs/release-v0.1.0.md` (release notes draft) ready. `pyproject.toml` version stays at `0.0.1` — Allen bumps to `0.1.0` on release.

### Test count progression
- Sprint 1 closeout: 23 passed
- Post Sprint 2 wave 1 (A): 27 passed
- Post Sprint 2 wave 2 (E): 48 passed
- Post Sprint 2 closeout (B + reverse + round-trip + schema check): **50 passed**

### Verification
```
$ uv run embodied-data convert data/agibot_sample/meta_info /tmp/v3 --from agibot --to lerobot-v3 --max-episodes 1
done: 375 frames, 1 task, 1 camera (observation.images.top_head) → /tmp/v3
$ uv run embodied-data convert /tmp/v3 /tmp/back --from lerobot-v3 --to agibot
done: 375 frames → meta_info/place_objects_into_handbag/<uuid> (+ sibling video) | fps=30
$ uv run embodied-data validate data/hf_v3_samples/pusht
Result: PASS
$ uv run embodied-data inspect data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-.../proprio_states.h5
(prints group/dataset tree)
$ uv build && uv run python -m twine check dist/*
PASSED (wheel + sdist)
```

### Blockers / open items
- HF_TOKEN [CRED] still absent — Beta access deferred to v0.1.1 (not blocking GA).
- Two subagent timeouts (Sprint 1 D / Sprint 2 B) on multi-file feature implementations. Pattern: timeout when subagent is asked to write 200+ LOC + tests in one go. Mitigation working: Tech Lead has authoritative schema docs already on disk, so direct takeover is ~15 min and merges cleanly. Worth keeping subagent prompts shorter going forward.
- 5 in-scope GitHub issue comment drafts in `docs/issue-comments-drafts.md` still pending Allen [PUBLISH] approval.
- v0.1 release: pending Allen [PUBLISH] approval to (1) push the `v0.1.0` tag and (2) `twine upload`. Plumbing is fully ready per `docs/release-checklist.md`.

### v0.2 candidates (do NOT pre-batch)
- Multi-camera (7 fisheye + hand) and depth (uint16 mm PNG bytes)
- True multi-episode-per-parquet rollover (Sprint 2 used one-episode-per-file simplification)
- Multi-episode reverse converter
- Cross-embodiment action retargeting (when third-party `lerobot-v3 → AgiBot` users actually appear)
- 2 remaining validator hardening items deferred from C's findings (full-corpus alignment, image-mode datasets)
- Generic `lerobot-v2 → lerobot-v3` upgrade (would address #2446 partially)

---

## Sprint 3 — 2026-04-30 (autonomous 6h sprint, 3 tracks)

### Track 1 ✅ — A.1 finalized: 4 in-scope upstream comments posted

After v0.1.0 GA shipped (PyPI + GitHub Release public), Allen approved second-round
post-v0.1.0 rewrites that reference `pip install embodied-data` + the release URL.
#124 was fully rewritten (stale "single-episode v0.0.2 / next sprint" wording
replaced with accurate v0.1.0 batch capability description). #2689 ALOHA stays
deferred to v0.2. Posted with audit-trail commits (one per issue):

| Issue | Comment URL | Commit |
| --- | --- | --- |
| AgiBot-World#18 | issuecomment-4356514656 | `c936f95` |
| AgiBot-World#124 | issuecomment-4356516845 | `9a244af` |
| AgiBot-World#149 | issuecomment-4356517466 | `7575672` |
| huggingface/lerobot#2158 | issuecomment-4356518120 | `2a602d7` |

### Track 2 ✅ — B: v0.1.1 patches landed (no release; awaiting [PUBLISH])

- **B.1 HF auth** ✅ — `user=allenwu06`. Alpha gating not approved for this
  user; Beta gating is approved. Pivoted to Beta (per-upstream-README schema is
  identical).
- **B.2 fixture acquired** ✅ — `data/agibot_beta_sample/675/936938/proprio_stats.h5`
  (1.17 MB) + `task_info_675.json` (417 KB). Beta proprio bundled in 784 MB tar
  (vs Alpha's 48 GB). Used `tarfile` streaming to extract just one h5 in <2 MB
  bandwidth. No source tar persisted.
- **B.3 / B.4 v0.1.0 vs real Beta** ✅ → `docs/v0.1.1-findings.md` (510 lines,
  15 triaged bugs across 5 BLOCKER + 8 HIGH + 2 LOW). Top deltas vs sim:
  filename `proprio_stats.h5` (no second `s`); `state/joint/position` shape
  `(N, 14)` not `(N, 34)` and zero attrs (vs sim's 34 joint names); `/timestamp`
  is `int64` ns Unix epoch not `float32` seconds; extra state subgroups
  `head` / `waist`; sparse `*/index` companions on most action subgroups;
  `task_info_<task>.json` is a list of episode dicts.
- **B.5 v0.1.1 patches** ✅ — 9 patches (8 from spec + 1 Tech Lead surfaced
  during verify) committed independently for audit trail:
  - Patch 1 `f7faeeb` — filename glob `proprio_stat[se]*.h5` accepts both names
    across 8 call sites; new `_agibot_paths.py` helper.
  - Patch 2 `45d52a5` — convert refuses real-robot input cleanly when
    `joint_dim != 34` or `attrs.name` missing.
  - Patch 3 `b7a12ae` — convert handles Beta `task_info_<task>.json` list shape.
  - Patch 4 `005e777` — preview reports actual joint dim, not always 22.
  - Patch 5 `c707ba1` — preview reads `robot_type` from h5 attrs (not hardcoded "a2d").
  - Patch 6 `453d770` — preview handles task_info list shape too (no more silent "(unknown)").
  - Patch 7 `4ccc0c3` — convert error path catches `KeyError` / `ValueError`,
    no more raw Rich tracebacks.
  - Patch 8 `537da4c` — `inspect` shows attrs per group/dataset (cheap diagnostic).
  - Patch 9 `e1c72f5` — batch `_discover_episodes` refuses Beta loudly instead
    of silently returning zero episodes (Tech Lead caught: B.fixer's
    `_assert_digitalworld_sim` worker concern was a symptom; root cause was
    upstream discovery never reaching the worker).
- **CHANGELOG ## [Unreleased]** staged (`2877d51`) listing the 7 user-visible
  v0.1.1 fixes + known limitations. **Version stays at 0.1.0**; bump + tag +
  twine upload all gated on Allen [PUBLISH].
- **Test count**: 50 → 64 (+14 v0.1.1 integration tests against the Beta
  fixture, all `@needs_beta` skipped on CI without the sample).
- **Explicitly v0.2** (untouched): bugs #3 (real Beta forward conversion),
  #5 (int64 ns timestamp arithmetic), #9 (head/waist/index mapping), #10
  (variable-length `*/index` companions), #11 (validate strict-mode for
  missing videos), #13 (error suggestion text differentiation), #15 (sim
  60Hz vs 30Hz doc inconsistency).

### Track 3 ✅ — C: v0.2 first feature scaffolded on draft PR

Selection: **real-Beta forward conversion**. Per Allen's autonomous-sprint
selection rule (B-driven > user-impact > issue-close), Track 2 surfaced 4
BLOCKER + 5 HIGH bugs that all map back to the same gap — v0.1 cannot
ingest real-robot AgiBot data. v0.1.1 shipped a refuse-and-document guard;
v0.2 lifts that guard.

Landed on `feat/v0.2-real-beta-ingest`:
- `docs/v0.2-real-beta-ingest.md` — 1-page design (scope, schema decisions
  including the `JOINT_14_BETA` best-guess, output 20-dim `observation.state`,
  test plan, out-of-scope, risks).
- `src/embodied_data/convert/agibot_beta_to_lerobot.py` — skeleton with
  `JOINT_14_BETA` + `OBSERVATION_STATE_NAMES_20` constants. Function raises
  `NotImplementedError` until subsequent commits implement.
- `tests/test_convert_beta.py` — 7 tests: 2 pin schema constants + skeleton
  refusal (passing), 5 integration tests `xfail(strict=True)` against the
  Beta fixture (each flips green when its impl path lands).

Draft PR: <https://github.com/allenwu-blip/embodied-data/pull/1>. TODO list
in PR body covers the implementation steps; merge gated on Allen [PUBLISH].

Test count: 50 → 66 passed (+16 v0.1.1 + v0.2 scaffold) + 5 xfailed.

### 6h sprint final report

| Track | Status | Commits | Tests Δ | Highlights |
| --- | --- | --- | --- | --- |
| Track 1 (post v0.1.0 issue comments) | ✅ done | 5 | 0 | #18, #124, #149, #2158 posted with audit trail |
| Track 2 (v0.1.1 patches) | ✅ done locally + pushed | 11 (9 patches + worklog + scaffold-fix) | +14 | filename glob, refuse-Beta, 8 silent-miscompile fixes |
| Track 3 (v0.2 scaffold) | ✅ done locally + draft PR | 1 (on feature branch) | +2 + 5 xfail | design + skeleton + draft PR #1 |

**Pending [PUBLISH]**:
- `## [Unreleased]` → `## [0.1.1]` bump + tag + GitHub release + twine upload (CHANGELOG already staged at `2877d51`).
- v0.2 PR #1 merge (gated on TODO checklist closure + Allen review).

**Pending [CRED]** (informational, not blocking):
- AgiBotWorld-Alpha gating not approved for `allenwu06`. Beta substituted; schemas identical per upstream README. If broader Alpha coverage is needed for v0.2 acceptance testing, request access at <https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha>.

**Two biggest discoveries**:
1. Sim DigitalWorld and real Beta diverge structurally — different filename
   (`proprio_state{s,}` typo persistent), different joint dim (34 vs 14),
   different timestamp dtype (`float32` sec vs `int64` ns), different state
   subgroups (Beta has `head`/`waist`), different `task_info` shape (single
   dict vs list of 399 dicts). v0.1's "DigitalWorld-only" framing held up
   under audit; v0.2 makes it explicit by splitting the converter modules
   and the schema docs.
2. Subagent timeout pattern remains real (Sprint 1 Subagent D, Sprint 2
   Subagent B, plus a near-miss this sprint where B.fixer's stated concern
   surfaced a 9th patch only after Tech Lead's verification). Mitigation
   continues to work: schema docs as durable ground truth, Tech Lead direct
   takeover when subagents stall.

**Top 3 priorities for next session**:
1. Resolve [PUBLISH] for v0.1.1 — bump pyproject + tag + twine upload (5 min, gated on user).
2. Implement v0.2 PR #1's TODO checklist — top of stack is `_agibot_paths` Beta layout detection + 20-dim observation builder. Each xfail test flips green incrementally.
3. Re-attempt Alpha access (Allen requests on HF) — gives a second real-data fixture variant for v0.2 acceptance + closes the "Alpha vs Beta schemas truly identical?" question empirically.

---

## v0.1.1 release + v0.2 PR #1 M1+M2 — 2026-04-30 (continuation)

### v0.1.1 GA ✅

Allen batch-approved the v0.1.1 7-step release flow. Sequence executed identically to v0.1.0:
- `## [Unreleased]` → `## [0.1.1]` rename + bump pyproject 0.1.0→0.1.1
- commit `ae3bb05` "chore: release v0.1.1 — sim/real-Beta compatibility patches"
- tag `v0.1.1` (annotated, single-line message), pushed
- tag CI run drift guard `version=0.1.1 tag=v0.1.1 match=true`
- draft GitHub release with prelude differential + CHANGELOG `[0.1.1]` body
- `rm -rf dist/ && uv build && twine check && twine upload && draft=false`
- PyPI live: <https://pypi.org/project/embodied-data/0.1.1/>
- GitHub release public: <https://github.com/allenwu-blip/embodied-data/releases/tag/v0.1.1>

CDN sync took ~70s; pip download fetched 38KB wheel byte-identical to local.

### v0.2 PR #1 — M1 + M2 landed on `feat/v0.2-real-beta-ingest`

**M1 ✅** — `feat(beta): convert AgiBot Beta single-episode → LeRobot v3` (`050579e`).
First v0.2 happy path. 14-dim joint + 2-dim effector + 2-dim head + 2-dim
waist → 20-dim `observation.state`; first-difference action; recomputed
`frame_index/30` timestamps; `task_info_<task>.json` list resolution at task
root; complete v3 dataset (no videos). All five xfail-pinned integration
tests flipped to passing.

**M2 ✅** — `feat(dispatcher): detect_agibot_variant routes sim vs Beta vs Alpha-stub`
(`b336972`). Adds `detect_agibot_variant(path) → {digitalworld, beta, alpha,
unknown}` + `schema_summary(path)` to `_agibot_paths.py`. Router in
`convert/__init__.py`: digitalworld → sim path; beta → Beta converter
(rejects batch flags — Beta batch is M3); alpha → friendly stub error
referencing PR #1; unknown → schema_summary in error. Two v0.1.1 patch
tests updated, one (Patch 7) explicitly skipped as superseded.

**Test count progression** on `feat/v0.2-real-beta-ingest`:
- Sprint 3 closeout: 66 passed + 5 xfailed
- Post-M1: 71 passed
- Post-M2: 86 passed + 1 skipped

### M3 deferred (next session)

Beta multi-episode batch (≥100 eps) needs ~150-300 LOC mirror of
`convert_agibot_batch`'s loop. Single-process happy path first cut.

### Top 3 priorities for next session (updated)

1. **M3 Beta batch** — primary v0.2 deliverable still outstanding. Real Beta
   tasks ship 100-400 episodes; without batch, M1's single-episode CLI is
   too narrow for v0.2 release notes.
2. **Schema doc reorg** — split `docs/schema-agibot.md` →
   `schema-agibot-{digitalworld,beta,overview}.md`. M2's friendly errors
   would be sharper with proper routing.
3. **Re-attempt Alpha access** (Allen-side, ~30 sec on HF) — empirical
   confirmation of "schemas identical per upstream README".

---

## Sprint 4 — 2026-04-30 / 05-01 (autonomous 6h, PR #1 ready-for-review)

Goal: drive PR #1 from "M1+M2 scaffold" to ready-for-review. Three tracks
all completed within budget.

### Track A ✅ — M3 Beta multi-episode batch (3 commits)

- `10d820a` `feat(beta): convert_agibot_beta_batch — multi-episode + resume + workers`.
  ~430 LOC mirroring `convert_agibot_batch`'s structure (sim) but stripped
  for Beta — no video plumbing, one-episode-per-file inside chunk-000.
  Includes `_BetaEpisodeSource`/`_BetaPerEpisodePayload` dataclasses,
  `_discover_beta_episodes`, `_load_beta_episode` (worker), `_commit_beta_episode`,
  resume helpers, multi-write helpers, `is_beta_batch_src` heuristic, and
  failed-episode logging to `<dst>/.beta_batch_errors.jsonl`.
- `e9ec7b2` `feat(dispatcher): route Beta task root → batch (M3 wiring)`.
  `detect_agibot_variant` now returns `beta` for Beta task roots
  (task_info_*.json sibling + proprio_stats.h5 in subtree). Dispatcher
  auto-batches when src is a task root or any of `--max-episodes` /
  `--resume` / `--workers` is set; rejects batch flags against a
  single-episode dir with a clear hint.
- `cb7ac6f` `test: M3 Beta batch — 10 integration tests`. 10-episode
  smoke (synthetic via symlinks), `--max-episodes` truncation, `--resume`
  idempotency, `--resume` fills only missing, corrupted h5 → error log,
  workers=1 vs workers=2 stats equivalence, multi-episode validate PASS,
  CLI dispatcher auto-routing.

Tech Lead direct-implementation per the documented "no subagent for ≥150
LOC implementation" mitigation. No timeout, no rework.

### Track B ✅ — Alpha empirical verification + schema doc reorg (2 commits)

- `494a700` `feat(dispatcher): route Alpha→Beta after empirical schema equivalence`.
  Alpha access landed (allenwu06 approved). Stream-extracted Alpha task 389
  / episode 656913 from upstream's 48 GB proprio tar (1.16 MB consumed).
  Three-way h5 diff (Alpha vs Beta vs Sim DigitalWorld) confirms upstream
  README's "schemas equivalent" claim: Alpha and Beta both have 14-dim
  joint float64, missing `state/joint.attrs["name"]`, int64 ns Unix-epoch
  timestamps, identical `state/{effector,end,head,waist,robot}` subgroups,
  sparse `*/index` companions on action subgroups. Dispatcher's `'alpha'`
  branch no longer stubs out — it prints a one-line "routing through Beta"
  note and falls through to the Beta branch.
- `413ea44` `docs(schema): reorg agibot schema by variant`. Split
  `docs/schema-agibot.md` (DigitalWorld-only at this point) into
  `docs/schema/{overview,digitalworld,beta}.md`. Overview includes the
  three-way diff table + B.1 verification + variant detection ladder +
  per-variant coverage matrix. Old `schema-agibot.md` becomes a stub
  redirect so legacy code/test/release-notes references still work.

### Track C ✅ — final tests + PR description + ready-for-review (2 commits)

- `5f8082b` `test: M3 + design §5 — no-video validate + sparse index drop`.
  Two more pinning tests:
  - `test_batch_no_video_dataset_passes_validate`: confirms `validate`
    handles a video-less Beta v3 (fps consistency SKIP, frame-video
    alignment PASS via metadata proxy).
  - `test_batch_does_not_leak_sparse_index_companions`: pins the v0.2
    policy that Beta `action/*/index` columns are dropped on conversion
    and never leak into `data/*.parquet` columns.
- PR #1 description rewritten with: "What this PR delivers" (5-bullet
  M1/M2/M3/Alpha/docs), "Try it" (six copy-paste commands covering sim,
  Beta single, Beta batch, Alpha), "Known gaps deferred to v0.3"
  (videos, sparse index, end-pose, reverse, joint-names override,
  per-frame raw timestamp), TODO checklist (M1-C all checked, "Allen
  review" + "v0.2.0 release sequence" the two open).
- PR title: "feat: v0.2 real-Beta forward conversion (scaffold)" →
  "feat: v0.2 real-Beta forward conversion (M1+M2+M3+Alpha+docs)".
- `gh pr ready 1` flipped from draft → ready-for-review.

### Sprint 4 final state

| Commits since v0.1.1 release | 8 (5 feature + 1 doc + 2 test) |
| Tests on `feat/v0.2-real-beta-ingest` | **98 passed + 1 skipped** |
| Ruff | clean |
| PR #1 | ready-for-review (no longer draft) |
| Branch additions / deletions | +2555 / -476 |
| Total session time | ~5h |

**[PUBLISH] queue (unchanged from Sprint 3 closeout)**:
- v0.2.0 release sequence — bump pyproject 0.1.1→0.2.0, tag v0.2.0,
  twine upload, GitHub release per `docs/release-checklist.md`. Gated on
  Allen review of PR #1 + merge to main.

**[CRED]** — all current credentials live (HF_TOKEN with Alpha + Beta
access; PyPI token in ~/.pypirc; `gh auth` as allenwu-blip).

### Top 3 priorities for next session

1. **Allen reviews PR #1** + merges to main (or asks for changes). Once
   merged, the v0.2.0 release sequence is mechanical.
2. **v0.2.0 release** — same 7-step `[PUBLISH]`-gated flow as v0.1.0/v0.1.1.
3. **Pick the next v0.3 feature** from the deferred list:
   videos / sparse index / end-pose / reverse-Beta / joint-names override.
   Top candidate by user-impact: videos for Beta (a real Beta dataset
   without video columns is much less useful for VLA fine-tuning).

---

## v0.2.0 GA — 2026-05-01

Allen squash-merged v0.2 PR #1 to main; v0.2.0 release ran the same 7-step
flow as v0.1.0 / v0.1.1.

- `5864816` `docs: capture v0.2.x patch backlog (task-name single-episode resolution)` — recorded the issue Allen flagged during release prep (Beta single-episode task-name resolver only walks `src` and `src.parent`, missing the canonical `task_info_<task>.json` at `src.parent.parent`; out of scope for v0.2.0, v0.2.1 candidate).
- `a564031` `chore: release v0.2.0 — Beta + Alpha forward conversion` — bump pyproject + `__version__` 0.1.1 → 0.2.0; CHANGELOG `[0.2.0]` section (Added: Beta single + batch + dispatcher + schema docs reorg; Changed: Alpha no longer refused — empirical equivalence verified; Known limitations deferred to v0.3+).
- Tag `v0.2.0` (annotated, single-line message), pushed; tag CI run `25202379310` ✓ success, drift guard `version=0.2.0 tag=v0.2.0 match=true`.
- Draft GitHub release built with prelude + Quick Start that does NOT assume any local fixture (uses public `lerobot/pusht` for sim path + Beta path with HF gating disclaimer + link to `docs/schema/beta.md` §1 for the stream-extract trick).
- `rm -rf dist/ && uv build && twine check && twine upload && gh release edit v0.2.0 --draft=false` — all green.
- PyPI live: <https://pypi.org/project/embodied-data/0.2.0/> (wheel 47.6 kB / sdist 35.5 kB; pip download fetched byte-identical wheel after a brief uv-index cache invalidation).
- GitHub release public: <https://github.com/allenwu-blip/embodied-data/releases/tag/v0.2.0>.

**Patch backlog appended** (recorded in `docs/v0.2.x-patches.md` for the next patch sprint, NOT shipping in v0.2.0):
- Beta single-episode CLI falls back to `task = "unknown"` (the ancestor-walk fix described above).
- Release-notes Beta Quick-Start uses literal `<episode_id>` placeholder — copy-paste-broken for external users; `gh release edit` fix when convenient.

---

## Total project timeline — sprint 1 brief → v0.2.0 GA

**Origin** (2026-04-24): Allen wanted to enter the embodied-AI data-tools
space as a one-person company. Two rounds of global research (CN + US robotics
companies + UMich Robotics labs + GitHub issue forensics) located the highest-
density pain point: AgiBot ↔ LeRobot v3 schema-conversion friction, with five
upstream issues already complained-about and unresolved. Decision: ship a
focused converter + validator, OSS-first, leverage Claude Code throughput.

**Cumulative metrics (counted on main as of 2026-05-01)**:

| Maxim | Number |
| --- | --- |
| Commits on main | **48** |
| Releases shipped (PyPI + GitHub) | **3** — v0.1.0, v0.1.1, v0.2.0 |
| Tags | `v0.1.0`, `v0.1.1`, `v0.2.0` |
| Tests | 0 → **98 passed + 1 skipped** |
| Lines of CHANGELOG | 0 → 100+ |
| Schema reference docs | 0 → 4 (`docs/schema-lerobot-v3.md` + `docs/schema/{overview,digitalworld,beta}.md`) |
| Upstream issue comments posted | **4** (AgiBot-World #18 / #124 / #149, huggingface/lerobot #2158) |
| Issue triage drafts on file | 10 (5 in-scope posted, 5 out-of-scope politely declined or deferred) |
| HF datasets validated end-to-end | 4 (`lerobot/pusht`, `lerobot/unitreeh1_warehouse`, `gpudad/so101_pick_cube_chunked`, `agibot-world/AgiBotWorld-Alpha` head-to-head) |
| Subagent dispatches | ~15 across all sprints (5 Sprint 1, 5 Sprint 2, 6 Sprint 3) |
| Subagent stalls / timeouts | 3 (Sprint 1 D — converter, Sprint 2 B — reverse, Sprint 2 B.fixer near-miss). All recovered via Tech-Lead-direct-implementation with schema docs as durable ground truth. |
| Sprints | 5 (Sprint 1 foundation; Sprint 2 v0.1 GA; Sprint 3 6h autonomous Track 1+2+3 → v0.1.1 + v0.2 scaffold + draft PR; Sprint 4 6h autonomous M3 + Alpha verify + schema reorg + ready-for-review; v0.2.0 release sprint) |

**Releases by content**:

- **v0.1.0** (Sprint 2 closeout, 2026-04-30): first public release. typer CLI, sim DigitalWorld → LeRobot v3 single-episode + batch, lerobot-v3 → sim AgiBot reverse, validate (5 checks), preview, inspect, --json, --version, schema docs, HF dataset survey. 50 tests.
- **v0.1.1** (Sprint 3 closeout, 2026-04-30): sim/real-Beta compatibility patches. Filename glob accepts both `proprio_state{s,}.h5`, list-of-episodes `task_info` resolution, preview honest joint-count reporting, robot_type from h5 attrs, error-path KeyError wrap, inspect attrs visibility, batch-discover refuses Beta loudly, v0.1's "refuse-and-document" guard for Beta. 64 tests.
- **v0.2.0** (this release, 2026-05-01): real-hardware Beta + Alpha forward conversion. Schema-detect dispatcher, Beta single + batch + reverse-stub, Alpha empirical equivalence + auto-route. 98 tests + 1 skipped.

**Subagent timeout pattern (durable lesson)**: subagents stall around 200+ LOC + tests in a single dispatch. Mitigation that worked all three times: schema documentation written to `docs/schema/` BEFORE implementation, so when the subagent timed out, Tech Lead resumed with full context and shipped equivalent code in 15-20 min. Going forward: keep subagent prompts under ~150 LOC of expected output; for bigger work, Tech Lead writes directly.

**Standing [PUBLISH] queue**: empty. All four categories of in-flight artifacts (PyPI releases, GitHub releases, issue comments, tags) are landed.

**Standing [CRED]**: all live (HF token with Alpha + Beta + DigitalWorld access, PyPI token in `~/.pypirc`, `gh auth` as `allenwu-blip`).

Next sprint direction is Allen-defined; no Tech Lead carry-over.

---

## Sprint 6 closeout — v0.3.0 head_color video (2026-05-01)

**Sprint goal**: ship `observation.images.head_color` for Beta single + batch so that LeRobot v3 datasets emitted by `embodied-data convert` are usable for VLA fine-tuning end-to-end. Other cameras and unrelated v0.3.x candidates strictly out of scope.

**Why it mattered**: v0.2.0 release notes' first Known Limitation — "Videos for Beta. v0.2 emits `video_path: null`" — was the largest user-facing blocker. Any lab seriously trialing the converter would notice the missing video on first run and bounce. v0.3 fixes that one thing.

**Track A — implementation (5 commits)**:

| Commit | Subject |
| --- | --- |
| `df1e7a0` | refactor(video): extract reencode_video + probe_video to shared `_video.py` |
| `56758ad` | feat(beta): emit `observation.images.head_color` video on single-episode convert |
| `d22c377` | feat(beta): emit head_color video on multi-episode batch convert |
| `accf03b` | feat(validate): hard-fail when declared video is missing or broken |
| `6d39707` | test(video): integration + unit + negative tests for Beta head_color pipeline |

**Track B — docs (1 commit)**: `7e29024` — CHANGELOG `## [Unreleased]` section, `docs/schema/beta.md` §10 video appendix, `docs/v0.3.x-patches.md` backlog file (multi-camera / sparse index / end-pose / reverse Beta), `scripts/fetch_beta_video_fixture.py` for reproducible test fixture acquisition.

**Track C — closeout (this commit)**: README Roadmap "v0.3 next" → "v0.3 shipped on `main`, awaiting tag", coverage block bumped 48 → 57 commits / 98 → 114 tests, this WORKLOG entry.

**Pytest**: 98 passed + 1 skipped → **114 passed + 1 skipped**, all green; ruff clean; v0.3.0 release flow not run (deliberately deferred — release decision is Allen's [PUBLISH] gate).

**Fixture acquisition decision**: Allen authorized a three-step descent ladder (stream-extract Beta → Alpha fallback → synthetic). Path 1 partially worked: `head_color.mp4` (8 MB, av1) stream-extracted cleanly via HTTP Range against `observations/675/880749-912853.tar` (36 GB tar, ~8 MB downloaded, ~10 s). Path 1 stalled on proprio: episode 882736's `proprio_stats.h5` lives in a 48 GB tar with non-numerically-sorted task blocks; HfFileSystem streaming hit ~3.5 entries/sec and didn't reach task 675 in the budgeted window. Decision: **real upstream video + 879-frame slice of the existing 936938 proprio for the proprio companion** (real schema, real Beta values, sliced to match video length). Path 2 (Alpha) and Path 3 (synthetic mp4) were not used. Documented explicitly in `scripts/fetch_beta_video_fixture.py` so the choice is reproducible and auditable.

**Critical design choice — video-before-data commit ordering**: original implementation wrote the per-episode parquet first, then re-encoded video, then wrote episode meta. End-to-end test on a mixed batch (1 episode with video + 1 without) revealed an orphaned `data/file-001.parquet` for the video-failed episode. Re-ordered commit to encode video FIRST so a missing/broken upstream mp4 fails the commit before any state-laden parquets land. Verified: failed episode produces zero on-disk state, succeeded episode stays clean, validate passes.

**Validate hard-fail update**: pre-v0.3, when `info.features` declared a video feature but episode-meta parquets lacked the corresponding `videos/<key>/from_timestamp` columns, the alignment check silently PASSed because the inner per-episode loop's `vid_keys` was empty. Tightened to four explicit FAIL paths: missing mp4 file on disk, missing episode-meta video columns, codec decode error, frame-count divergence >1 frame. Proprio-only output (no `dtype: video` in features) still SKIPs cleanly.

**No subagent timeouts this sprint**. Tech Lead direct implementation throughout per durable Sprint 1-3 lesson — Beta video integration was ~250+ LOC across 3 files, well above the ~150 LOC subagent-timeout threshold. Tests (340 LOC) could have been a subagent dispatch but were also Tech Lead direct since the pattern was small and the Track A implementation was already in-context.

**Sprint metrics (cumulative through Sprint 6)**:

| Maxim | Was (v0.2.0 GA) | Now (v0.3.0 staged) |
| --- | --- | --- |
| Commits on main | 48 | **57** |
| PyPI releases | 3 (0.1.0/0.1.1/0.2.0) | 3 (v0.3.0 staged in `## [Unreleased]`) |
| Tests passing | 98 + 1 skipped | **114 + 1 skipped** (+16) |
| HF datasets exercised | 4 | 4 (same — Beta video re-uses existing access) |
| Sprints | 5 | 6 |

**Standing [PUBLISH] queue**: 1 item — v0.3.0 release sequence (CHANGELOG → bump version → tag → twine → `gh release`). Staged in `## [Unreleased]`, awaits Allen's manual trigger per the v0.1.0 / v0.1.1 / v0.2.0 pattern.

**Standing [CRED]**: all live (no rotation needed since v0.2.0).

**Reminder for Allen** — v0.3.0 ship was the milestone Allen pre-committed to as the cowork distribution-timing recheck point. When v0.3.0 is tagged, that conversation should be reopened (which channels to push, which upstream issue threads to update, whether any current "no external distribution" guardrails should lift).

**Next sprint candidates (top 3, Allen-decided priority)**:

1. **v0.3.1 multi-camera** — fisheye / hand / back. Same h264 contract per camera, generalize `find_head_color_video` → `find_camera_videos(ep_dir) -> dict[str, Path]`. Largest single user-facing gap remaining for VLA training.
2. **v0.3.2 sparse `*/index` masks** as `auxiliary.<group>.mask` features. Smaller surface; relevant for imitation-learning pipelines that need teleoperator-active timesteps.
3. **Real-user feedback channel**. v0.3.0 makes the converter usable end-to-end; the next pipeline-rate-limiter is whether labs are actually trying it. Options: lurk Discord/Slack robotics communities, post a "v0.3.0 real video" comment on the upstream issue threads we're already on, or ask UMich Robotics labs directly.
