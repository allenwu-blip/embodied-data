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
