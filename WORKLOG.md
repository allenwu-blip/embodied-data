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
