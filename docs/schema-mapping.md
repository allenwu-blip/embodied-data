# AgiBot → LeRobot v3 Field Mapping

Authoritative field-by-field mapping for the v0.1 converter. Source schemas:
[`docs/schema-agibot.md`](./schema-agibot.md) for the AgiBot side and
[`docs/schema-lerobot-v3.md`](./schema-lerobot-v3.md) for the LeRobot v3
side. Citations format: `script:LINE` = `data/agibot_sample/scripts/convert_to_lerobot.py`,
`v3:§N` = `docs/schema-lerobot-v3.md` section N, `agibot:§N` = `docs/schema-agibot.md` section N.

---

## 1. Per-frame tabular fields (→ `data/chunk-000/file-000.parquet`)

| LeRobot v3 column | Source HDF5 path | Source dtype/shape (post-subselect) | Required transform | Provenance | Status |
|---|---|---|---|---|---|
| `observation.state` | `state/joint/position[:, idx_22]` | float32 (375, 22) | (1) read 34-col matrix; (2) subselect 22 cols by joint name in the order `[head_yaw, head_pitch, lift_body, body_pitch, J1_l, J1_r, J2_l, J2_r, …, J7_l, J7_r, right_Left_1, right_Right_1, left_Left_1, left_Right_1]`; (3) cast to float32 (already is). Units: rad → rad (no-op). | `script:243-308`, `v3:§5` | verified on this sample |
| `action` | `state/joint/position[:, idx_22]` (NOT `action/joint/position`) | float32 (375, 22) | First-difference: `action[i] = state[i+1] − state[i]` for `i ∈ [0, N-2]`; pad with copy of last delta to length N. **Do not read `action/joint/position` — it is identical to state in this sample (`agibot:§2.2`).** | `script:293-302` | verified on this sample |
| `timestamp` | (synthesized) | float32 (375,) | `frame_index / 30.0`. Do **not** copy AgiBot's `/timestamp` dataset — its values imply 60 Hz which contradicts the mp4 30 Hz (`agibot:§5`). | `v3:§5` ("`timestamp = frame_index / fps`") | verified |
| `frame_index` | (synthesized) | int64 (375,) | `np.arange(N)` per episode, 0-based | `v3:§5` | verified |
| `episode_index` | constant | int64 (375,) | `0` for single-episode v0.1 | `v3:§5` | verified |
| `index` | (synthesized) | int64 (375,) | global frame counter, monotonically increasing across episodes; for one-episode = `frame_index` | `v3:§5` | verified |
| `task_index` | from `task_info.json.task_name` | int64 (375,) | resolve via `meta/tasks.parquet`; for single-episode single-task = `0` | `v3:§4` | verified |

**LeRobot `info.features` declarations** (per `v3:§2.1`):

```jsonc
"observation.state": {"dtype":"float32","shape":[22],"names":[<22 joint names in order above>]},
"action":            {"dtype":"float32","shape":[22],"names":[<same 22 names>]}
```

Both have feature_dim = 22, matching `script:150-157`. The `names:` array
must encode the order from `agibot:§3` exactly; otherwise a downstream policy
will silently swap left/right hands (the official script's effector
sub-order is right-then-left, see `agibot:§3`).

---

## 2. Camera mapping (RGB videos → `videos/<key>/chunk-000/file-000.mp4`)

| Source mp4 | LeRobot v3 feature_key | Source resolution | v3 `info.features` shape (CHW per `v3:§2`) | Transform |
|---|---|---|---|---|
| `video/head.mp4` | `observation.images.top_head` | 640×480 hevc 30fps | `[3, 480, 640]` | re-encode to libsvtav1 `g=2 crf=30 yuv420p` (`v3:§6`); preserve resolution |
| `video/hand_left.mp4` | `observation.images.hand_left` | 640×480 | `[3, 480, 640]` | same |
| `video/hand_right.mp4` | `observation.images.hand_right` | 640×480 | `[3, 480, 640]` | same |
| `video/head_front_fisheye.mp4` | `observation.images.head_center_fisheye` | 960×768 | `[3, 768, 960]` | same; **note 768, not 748** (`agibot:§6` bug 4) |
| `video/head_left_fisheye.mp4` | `observation.images.head_left_fisheye` | 960×768 | `[3, 768, 960]` | same |
| `video/head_right_fisheye.mp4` | `observation.images.head_right_fisheye` | 960×768 | `[3, 768, 960]` | same |
| `video/back_left_fisheye.mp4` | `observation.images.back_left_fisheye` | 960×768 | `[3, 768, 960]` | same |
| `video/back_right_fisheye.mp4` | `observation.images.back_right_fisheye` | 960×768 | `[3, 768, 960]` | same |

Provenance: feature names taken verbatim from `script:48-148`. `[C, H, W]`
order forced by `v3:§2` (overrides upstream's `[H, W, C]` declaration).

**fps:** all 8 streams confirmed at 30.0 fps (`agibot:§5`, verification A).
LeRobot v3 single `info.fps = 30` is sufficient. No resampling.

**Re-encoding:** unconditional. Even though the source is already hevc 30 fps,
`v3:§6 / F1` warns that reusing upstream timestamps causes
`av.error.ValueError [Errno 22]` during multi-episode concatenation. Always
re-encode each episode mp4 from scratch with `g=2`.

---

## 3. Depth mapping (PNGs → embedded image bytes)

| Source PNG | LeRobot v3 feature_key | Source dtype | v3 shape | Transform |
|---|---|---|---|---|
| `depth/<i>/head.png` | `observation.images.cam_top_depth` | uint16 (480, 640) | `[1, 480, 640]` (CHW; `v3:§5` says `image` features stored as embedded PNG bytes) | (1) read uint16 mm; (2) per `script:233`, divide by 1000 → float32 meters; (3) optionally encode as 16-bit grayscale PNG to preserve precision; (4) v3 `dtype: "image"` (NOT `"video"`) so bytes embed in parquet |
| `depth/<i>/hand_left.png`, `depth/<i>/hand_right.png` | — | — | — | **skip** — official script ignores (`script:316`); out of v0.1 scope (`agibot:§4.2`) |

Provenance: `script:62-65, 233, 316`.

**Open question (D):** `script:233` divides by 1000 to produce `float32`
meters, but v3 image features are typed as PNG bytes (`v3:§5`), so we'd be
re-encoding a float32 array to PNG which is lossy. Either (a) preserve the
raw uint16 mm PNG and document the unit in `feature.names`, or (b) accept
the precision loss. **Recommended: keep raw uint16 mm PNGs, store as
embedded image bytes; document units in `info.features[...].info`.**
Defer final call to D.

---

## 4. Auxiliary metadata mappings

### 4.1 Task → `meta/tasks.parquet`

| LeRobot column | Source | Transform | Provenance |
|---|---|---|---|
| index `task` (str) | `task_info.json.task_name` | use raw string `"place_objects_into_handbag"` (or replace `_` with space, defer) | `v3:§4` |
| `task_index` (int64) | n/a | `0` for single-task | `v3:§4` |

### 4.2 Episode metadata → `meta/episodes/chunk-000/file-000.parquet`

| LeRobot column | Source | Value (this sample) |
|---|---|---|
| `episode_index` | constant | `0` |
| `tasks` | from `task_info.json` | `["place_objects_into_handbag"]` |
| `length` | h5 frame count | `375` |
| `meta/episodes/{chunk_index,file_index}` | self | `0, 0` |
| `data/{chunk_index,file_index}` | self | `0, 0` |
| `dataset_from_index`, `dataset_to_index` | computed | `0, 375` |
| `videos/<vid_key>/{chunk_index,file_index}` | per camera | `0, 0` for each of 8 |
| `videos/<vid_key>/from_timestamp` | computed | `0.0` (first episode in mp4) |
| `videos/<vid_key>/to_timestamp` | computed | `375 / 30 = 12.5` |
| `stats/<feature>/...` | aggregated | per `v3:§3` |

### 4.3 `info.json` top-level

| Field | Value | Source |
|---|---|---|
| `codebase_version` | `"v3.0"` | `v3:§2` (mandatory) |
| `fps` | `30` | `agibot:§5` (verified) |
| `total_episodes` | `1` | this sample |
| `total_frames` | `375` | h5 row count |
| `total_tasks` | `1` | task_info.json |
| `robot_type` | `"a2d"` (or `"A2D_fixed"`) | `script:368`, h5 attr at `state/robot.attrs['name']` |
| `chunks_size`, `data_files_size_in_mb`, `video_files_size_in_mb` | defaults `1000, 100, 200` | `v3:§1` |
| `data_path`, `video_path` | defaults | `v3:§1` |

### 4.4 Subtasks (optional, deferred)

`task_info.json.label_info.action_config` carries `(start_frame, end_frame,
skill)` tuples — natural fit for `meta/subtasks.parquet` per `v3:§1`. Not
in v0.1 scope.

### 4.5 Camera intrinsics / extrinsics

`parameter.json` has rich calibration. **No canonical v3 slot** (`v3:§2`).
Drop for v0.1; could sidecar later under `meta/extra/`.

---

## 5. Field-mapping summary table (one-glance view)

| AgiBot (HDF5) | LeRobot v3 | Used by v0.1? |
|---|---|---|
| `state/joint/position[:, 22 joints]` | `observation.state` | YES |
| `state/joint/position[:, 22 joints]` (first-diff) | `action` | YES |
| 8 mp4s in `video/` | `observation.images.<8 keys>` | YES |
| `depth/<i>/head.png` | `observation.images.cam_top_depth` | YES |
| `task_info.json.task_name` | `meta/tasks.parquet` + `task_index` | YES |
| `/timestamp` (60 Hz, suspect) | discarded; recomputed as `i / 30` | NO (replaced) |
| `action/joint/position` (== state) | discarded | NO (redundant) |
| `state/effector/position` (mm) | discarded (effector covered by joint cols) | NO |
| `state/end/*`, `action/end/*` | — | NO (out of scope) |
| `state/robot/*`, `action/robot/*` | — | NO (out of scope) |
| `state/joint/{velocity,effort,current_value}` | — | NO |
| `depth/<i>/{hand_left,hand_right}.png` | — | NO |
| `parameter.json` (calibration) | — | NO |
| `task_info.json.label_info.action_config` | (would map to subtasks.parquet) | NO |

---

## 6. Open questions for D (Converter Builder)

1. **Depth encoding precision.** Keep source uint16 mm PNG bytes verbatim or
   convert to float32 meters PNG (`script:233` divides by 1000)?
   **Recommendation: keep uint16 mm bytes**, document `unit: "mm"` in
   `info.features["observation.images.cam_top_depth"].info`. Avoids precision
   loss; matches LeRobot v3's "embedded PNG bytes" model (`v3:§5`).

2. **Joint-name string for `names:` array.** The 22 names from
   `agibot:§3` come from upstream URDF; should we relabel to a more
   readable scheme (e.g. `left_arm_j1` instead of `Joint1_l`)? **Recommendation:
   keep upstream names verbatim.** Provenance is preserved; renaming risks
   a silent drift if upstream changes URDF naming.

3. **`robot_type` string.** Upstream script literal is `"a2d"`
   (`script:368`); h5 attribute is `"A2D_fixed"`; README mentions A2D variants
   without specifying. **Recommendation: use `"a2d"`** to match upstream
   convention; emit `robot_type_detail: "A2D_fixed"` as a comment in
   `info.json` (v3 `DatasetInfo.from_dict` ignores unknown fields per
   `v3:§2`, so this is non-fatal forward-compat).

---

## 7. Verification status legend

- "verified on this sample" — empirically confirmed by inspecting
  `proprio_states.h5` and the 8 mp4s in this single sample.
- "documented but not tested" — derived from upstream README + script,
  not yet executed end-to-end.
- "uncertain" — flagged for D to resolve at implementation time.

All rows above are "verified on this sample" unless noted. Multi-episode
behavior, multi-task batches, and all out-of-scope rows are "documented but
not tested".
