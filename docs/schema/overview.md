# AgiBot schema — overview & variant detection

This dataset family ships in three variants. They share the top-level
`state/...` and `action/...` group structure but diverge on joint
dimensionality, attribute presence, timestamp encoding, and a couple of
subgroups. `embodied-data` detects the variant from on-disk content and
routes to the right converter.

| Variant | Source | proprio filename | joint dim | attrs.name | timestamp | head/waist subgroups | sparse `*/index` companions |
|---|---|---|---|---|---|---|---|
| **DigitalWorld** | `agibot-world/AgiBotDigitalWorld` (sim) | `proprio_states.h5` (with `e`) | **34** | **present** (34 names) | float32 sec | absent | absent |
| **Beta** | `agibot-world/AgiBotWorld-Beta` (real) | `proprio_stats.h5` (no `e`) | **14** | **missing** | int64 ns Unix-epoch | present | present (action only) |
| **Alpha** | `agibot-world/AgiBotWorld-Alpha` (real) | `proprio_stats.h5` | **14** | **missing** | int64 ns Unix-epoch | present | present | 

## Empirical equivalence: Alpha ≡ Beta

The upstream README claims Alpha and Beta share schemas. Verified on
**2026-04-30** by extracting one episode from each and diffing
`f.visititems` output + dataset shapes/dtypes:

- Alpha: `task 389 / episode 656913`, 1430 frames, 14-dim joint float64.
- Beta: `task 675 / episode 936938`, 1090 frames, 14-dim joint float64.

Identical: top-level group tree, all dataset dtypes, presence of
`state/{joint,effector,end,head,waist,robot}` and matching action subgroups,
absence of `state/joint.attrs["name"]` on both, int64 ns timestamp encoding,
sparse `*/index` companions on action subgroups.

**Consequence**: the v0.2 Beta converter handles real Alpha data without
code changes. `convert/__init__.py` routes `variant=='alpha'` (currently
identified by an "alpha" component in the path) through the Beta converter
with a one-line console note.

## Per-variant references

- [DigitalWorld (sim)](./digitalworld.md) — full HDF5 tree, joint
  subselection (34→22), 8-camera + depth layout, 60 Hz timestamp upstream
  bug, 8 known bugs in upstream `convert_to_lerobot.py`.
- [Beta (real, also Alpha)](./beta.md) — 14-joint layout, head/waist
  subgroups, sparse `*/index` companions, int64 ns timestamps.

## Detection logic (canonical)

`embodied_data._agibot_paths.detect_agibot_variant(path)` returns one of
`digitalworld | beta | alpha | unknown`:

1. **Alpha hint**: any path component containing `alpha` (case-insensitive).
2. **DigitalWorld**: `proprio_states.h5` directly inside `path` *and* h5 has
   `state/joint/position` shape `(N, 34)` *and* `state/joint.attrs["name"]`
   present.
3. **Beta single-episode**: `proprio_stats.h5` directly inside `path` *and*
   h5 has `state/joint/position` shape `(N, 14)` *and* `attrs["name"]` is
   missing.
4. **Beta task root**: `task_info_*.json` directly inside `path` *and* at
   least one `proprio_stats.h5` somewhere in the subtree.
5. **DigitalWorld batch root**: at least one `proprio_states.h5` somewhere
   in the subtree.
6. Otherwise → `unknown` (the dispatcher emits a `schema_summary`
   description of what was actually found).

When two heuristics could fire (e.g., a path containing "alpha" *and* a
DigitalWorld-shaped h5), Alpha wins because the dispatcher routes it
through the Beta-equivalent converter. If you have a DigitalWorld path that
happens to contain "alpha" in its name and you do *not* want this routing,
rename the directory.

## Coverage by `embodied-data convert`

| Detected variant | v0.1 / v0.1.1 | v0.2 (PR #1) |
|---|---|---|
| DigitalWorld single-episode | ✅ | ✅ |
| DigitalWorld batch | ✅ | ✅ |
| Beta single-episode | ❌ refused | ✅ |
| Beta task root (batch) | ❌ silently skipped | ✅ |
| Alpha (any) | ❌ refused | ✅ (via Beta path) |
| Unknown | structured error w/ schema summary | structured error w/ schema summary |
