# Maintenance Mode

embodied-data is in maintenance mode as of 2026-05-02.

## What this means

- No new major features. The roadmap items previously listed for v0.4 (ALOHA HDF5 ingest, multi-camera, sparse `*/index`, end-pose flatten, cross-embodiment retargeting, RLDS/OpenX support) are NOT actively planned.
- Patch releases (0.3.x) will continue when bugs are reported or PRs are merged.
- Nightly CI runs and stays green.
- Issues, Discussions, and PRs are responded to within 24-48 hours.
- Schema / data validation is considered stable; behavioral changes are unlikely without a strong external signal.

## Why

The project shipped 5 PyPI releases (v0.1.0 → v0.3.1) covering AgiBot World ↔ LeRobot v3 (sim DigitalWorld + real Beta + Alpha) end-to-end with video pipeline. The maintainer is shifting focus to a new product (an open VLA evaluation leaderboard) where the same schema/validate machinery serves as backend infrastructure.

## What still works

- `pip install embodied-data` — fully functional, supported
- `convert` / `validate` / `preview` / `inspect` — all behaviors documented and stable
- All 5 schema reference docs (`docs/schema/`) — still accurate
- The 5 in-scope upstream issue threads (AgiBot-World #18 / #124 / #149, huggingface/lerobot #2158, #2689 deferred) — comments stand

## How to contribute

- File issues for bugs (the issue templates are wired up)
- Open PRs for fixes (the PR template includes the contributor checklist)
- Use Discussions for questions / use cases / feature ideas — note that "feature ideas" are now logged for community consideration, not for me to implement single-handed

## Re-activation triggers

I (the maintainer) will resume active development on embodied-data if any of the following happens:

- A real production user reports a blocking bug
- A security vulnerability or dependency CVE
- The new product (leaderboard) reaches a maturity point where feature work on embodied-data unlocks a leaderboard feature
- A strategic shift makes embodied-data the right primary focus again
