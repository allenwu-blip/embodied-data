# AgiBot DigitalWorld — HDF5 Schema Reference

Ground-truth schema for the embodied-data v0.1 converter, captured from the
single sample at
`data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96/`
(task `place_objects_into_handbag`, 375 H5 frames). All claims are pinned to
either the upstream README at `data/agibot_sample/README.md` (line numbers
below as `README:LINE`) or the canonical conversion script
`data/agibot_sample/scripts/convert_to_lerobot.py` (`script:LINE`).

The actual proprio file on disk is named **`proprio_states.h5`** — the
upstream README at `README:259` calls it `proprio_stats.h5` and the official
conversion script at `script:243` opens `aligned_joints.h5`. Both names are
wrong for the sample as released. Confirmed by directory listing.

---

## 1. Top-level group tree (`f.visititems`)

```
G  action
G  action/effector
D  action/effector/force                 shape=(375,)        dtype=float32
D  action/effector/index                 shape=(375,)        dtype=float32
D  action/effector/position              shape=(375, 2)      dtype=float32
G  action/end
D  action/end/angular                    shape=(0,)          dtype=float32
D  action/end/orientation                shape=(375, 2, 4)   dtype=float32
D  action/end/position                   shape=(375, 2, 3)   dtype=float32
D  action/end/velocity                   shape=(0,)          dtype=float32
D  action/end/wrench                     shape=(0,)          dtype=float32
G  action/joint
D  action/joint/current_value            shape=(0,)          dtype=float32
D  action/joint/effort                   shape=(375, 34)     dtype=float32
D  action/joint/position                 shape=(375, 34)     dtype=float32
D  action/joint/velocity                 shape=(375, 34)     dtype=float32
G  action/robot
D  action/robot/orientation              shape=(375, 4)      dtype=float32
D  action/robot/velocity                 shape=(375,)        dtype=float32
G  state
G  state/effector
D  state/effector/force                  shape=(375,)        dtype=float32
D  state/effector/index                  shape=(375,)        dtype=float32
D  state/effector/position               shape=(375, 2)      dtype=float32
G  state/end
D  state/end/angular                     shape=(0,)          dtype=float32
D  state/end/orientation                 shape=(375, 2, 4)   dtype=float32
D  state/end/position                    shape=(375, 2, 3)   dtype=float32
D  state/end/velocity                    shape=(0,)          dtype=float32
D  state/end/wrench                      shape=(0,)          dtype=float32
G  state/joint
D  state/joint/current_value             shape=(0,)          dtype=float32
D  state/joint/effort                    shape=(375, 34)     dtype=float32
D  state/joint/position                  shape=(375, 34)     dtype=float32
D  state/joint/velocity                  shape=(375, 34)     dtype=float32
G  state/robot
D  state/robot/orientation               shape=(375, 4)      dtype=float32
D  state/robot/orientation_drift         shape=(375, 4)      dtype=float32
D  state/robot/position                  shape=(375, 3)      dtype=float32
D  state/robot/position_drift            shape=(375, 3)      dtype=float32
D  timestamp                             shape=(375,)        dtype=float32
```

No file-level attributes. Group attributes:

| Group | Attribute | Type | Notes |
|---|---|---|---|
| `state/joint`, `action/joint` | `name` | object array (Python `str`), shape `(34,)` | ordered joint names. **`np.str_`, not bytes — no `.decode()`.** |
| `state/effector`, `action/effector` | `name` | `(2,) object` | `['left', 'right']` |
| `state/effector`, `action/effector` | `category` | `(1,) object` | `['continuous']` |
| `state/end`, `action/end` | `name` | `(2,) object` | `['left', 'right']` |
| `state/robot`, `action/robot` | `name` | `(1,) object` | `['A2D_fixed']` (robot type) |

Datasets shown with `shape=(0,)` are empty placeholders kept for schema
stability (README:326-327: "*effort* and *wrench* not available for now").

---

## 2. Per-group field reference

### 2.1 `state/joint` — proprioceptive joint states

| Field | Shape | Dtype | Units | Range (this sample) | Used by v0.1? |
|---|---|---|---|---|---|
| `position` | (375, 34) | float32 | radians | `[-2.739, 2.892]` | **Yes** — subselect 22 cols → `observation.state` |
| `velocity` | (375, 34) | float32 | rad/s | n/a verified | No (out of v0.1 scope) |
| `effort`   | (375, 34) | float32 | N·m | placeholder | Ignore (README:326: "Not available for now") |
| `current_value` | (0,) | float32 | — | empty | Ignore |

Group attr `name` (34 strings) — see §3 for ordering.

### 2.2 `action/joint` — commanded joints

Same five datasets as `state/joint` plus `effort` and `velocity`. **Critical
finding:** in this sample `action/joint/position == state/joint/position`
elementwise (max abs diff = `0.0`, verified). The official converter therefore
synthesizes a delta-action via first-difference of state (`script:293-302`)
rather than reading `action/joint/position` directly. Until proven otherwise
on other tasks, treat `action/joint/position` as **redundant with state** and
prefer the first-difference recipe.

### 2.3 `state/effector` / `action/effector` — gripper aperture

| Field | Shape | Dtype | Units | Range | Used by v0.1? |
|---|---|---|---|---|---|
| `position` | (375, 2) | float32 | mm (gripper opening) | `[119.28, 239.94]` | Already covered by 4 effector joints in §2.1; **ignore** here |
| `force`    | (375,)   | float32 | N | all 0 in this sample | Ignore (placeholder) |
| `index`    | (375,)   | float32 | frame counter | `[0, 374]` | Ignore (control-source bookkeeping) |

`state/effector/position` is **mm**, while the four effector entries inside
`state/joint/position` (`{left,right}_{Left,Right}_1_Joint`) are **radians**
in `[-1.0, 1.0]`. The official script uses the radian joint columns, not the
mm aperture — we follow that convention.

### 2.4 `state/end` / `action/end` — bimanual end-effector pose (TCP)

| Field | Shape | Dtype | Units / Convention |
|---|---|---|---|
| `position` | (375, 2, 3) | float32 | xyz meters, world frame; `[:, 0]` = left, `[:, 1]` = right |
| `orientation` | (375, 2, 4) | float32 | quaternion **wxyz** (README:334) |
| `angular`, `velocity`, `wrench` | (0,) | float32 | empty placeholders |

Out of scope for v0.1 (LeRobot v3 has no canonical `observation.end_pose`
key; would have to be encoded as a flat vector under `observation.state`,
which conflicts with the joint-position semantics). Defer.

### 2.5 `state/robot` / `action/robot` — robot base in world

| Field | Shape | Dtype | Notes |
|---|---|---|---|
| `state/robot/position` | (375, 3) | float32 | xyz, meters (z always 0 per README:340) — first row `[12.95, 12.30, 0.10]` |
| `state/robot/orientation` | (375, 4) | float32 | wxyz quaternion |
| `state/robot/orientation_drift`, `position_drift` | (375, …) | float32 | sim-only odometry drift terms; ignore |
| `action/robot/orientation` | (375, 4) | float32 | commanded base orientation |
| `action/robot/velocity` | **(375,)** | float32 | **README:348 promises (N, 2) `[v_x, yaw_rate]` but this sample is 1-D `(375,)`. All zeros in this sample.** Schema drift — see §6. |

Out of scope for v0.1 (mobile-base actions are not part of the 22-dim
manipulation policy the official converter targets).

### 2.6 `timestamp`

| Shape | Dtype | Units | First / Last | Implied rate |
|---|---|---|---|---|
| (375,) | float32 | seconds (sim time, README:332) | `0.20833` … `6.44167` | 60.0 Hz (period 0.01666…s) |

Strictly monotonically increasing (verified). Period drift std = 0.000000;
this is synthetic sim data, not real-robot capture, so monotonicity is
guaranteed. **The implied 60Hz contradicts the 30 fps mp4 streams in §4.**
See §5.

---

## 3. Joint-name list (34) and the official 22-subset

Order on disk (`state/joint.attrs['name']`):

| Idx | Name | Kept by official script? | Group |
|---|---|---|---|
|  0 | `joint_lift_body` | yes | body |
|  1 | `joint_body_pitch` | yes | body |
|  2 | `joint_head_yaw` | yes | head |
|  3 | `joint_head_pitch` | yes | head |
|  4 | `Joint1_l` | yes | left arm |
|  5 | `Joint1_r` | yes | right arm |
|  6 | `Joint2_l` | yes | left arm |
|  7 | `Joint2_r` | yes | right arm |
|  8 | `Joint3_l` | yes | left arm |
|  9 | `Joint3_r` | yes | right arm |
| 10 | `Joint4_l` | yes | left arm |
| 11 | `Joint4_r` | yes | right arm |
| 12 | `Joint5_l` | yes | left arm |
| 13 | `Joint5_r` | yes | right arm |
| 14 | `Joint6_l` | yes | left arm |
| 15 | `Joint6_r` | yes | right arm |
| 16 | `Joint7_l` | yes | left arm |
| 17 | `Joint7_r` | yes | right arm |
| 18 | `left_Left_1_Joint` | yes | left effector |
| 19 | `left_Right_1_Joint` | yes | left effector |
| 20 | `right_Left_1_Joint` | yes | right effector |
| 21 | `right_Right_1_Joint` | yes | right effector |
| 22 | `left_Left_0_Joint` | **dropped** | underactuated finger |
| 23 | `left_Left_Support_Joint` | **dropped** | passive support link |
| 24 | `left_Right_0_Joint` | **dropped** | underactuated finger |
| 25 | `left_Right_Support_Joint` | **dropped** | passive support link |
| 26 | `right_Left_0_Joint` | **dropped** | underactuated finger |
| 27 | `right_Right_0_Joint` | **dropped** | underactuated finger |
| 28 | `Left_Left_RevoluteJoint` | **dropped** | passive revolute |
| 29 | `Left_Right_RevoluteJoint` | **dropped** | passive revolute |
| 30 | `right_Left_Support_Joint` | **dropped** | passive support link |
| 31 | `right_Right_Support_Joint` | **dropped** | passive support link |
| 32 | `right_Left_RevoluteJoint` | **dropped** | passive revolute |
| 33 | `right_Right_RevoluteJoint` | **dropped** | passive revolute |

Why drop 12: the gripper at each fingertip on this A2D platform has 1 driven
joint plus 5 mimic / passive / support joints per side. Only the `*_1_Joint`
names are commanded (`script:272-277`); the others are determined by the
mimic kinematics. Dropping them reduces the policy dimension from 34 to 22
without information loss for behavior cloning.

**Order chosen by the official script** (`script:306-308`,
`np.hstack([head, body, arm, effector])`):

```
[head_yaw, head_pitch,           # idx 0..1   (raw idx 2,3)
 lift_body, body_pitch,          # idx 2..3   (raw idx 0,1)
 J1_l, J1_r, J2_l, J2_r,         # idx 4..17  (raw idx 4..17)
 J3_l, J3_r, J4_l, J4_r,
 J5_l, J5_r, J6_l, J6_r,
 J7_l, J7_r,
 right_Left_1, right_Right_1,    # idx 18..21 (raw idx 20,21)
 left_Left_1,  left_Right_1]     # idx 22..23?? — see below
```

**Subtle bug in the official ordering** (`script:272-277`): the effector
sublist is `[right_Left_1, right_Right_1, left_Left_1, left_Right_1]` — i.e.
**right hand first, then left hand**, while the arm sublist is left-then-right
interleaved. The order is consistent in itself but unusual; downstream
`names: [...]` strings on the LeRobot v3 feature must encode this order
faithfully or else a policy trained on this data will swap hands silently.

Per-joint ranges in this sample (selected): head_yaw fixed at 0; lift_body
fixed at 0.1998; body_pitch ≈ 0.5; arms span up to ±2.74; effector aperture
joints in `[-1, +1]` rad.

---

## 4. Multi-modal layout

### 4.1 RGB cameras (8) — `observation/<uuid>/video/*.mp4`

| File | LeRobot key (per `script:48-148`) | Resolution | Codec | Pix fmt | nb_frames | Avg rate | Duration |
|---|---|---|---|---|---|---|---|
| `head.mp4` | `observation.images.top_head` | 640×480 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `hand_left.mp4` | `observation.images.hand_left` | 640×480 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `hand_right.mp4` | `observation.images.hand_right` | 640×480 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `head_front_fisheye.mp4` | `observation.images.head_center_fisheye` | 960×768 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `head_left_fisheye.mp4` | `observation.images.head_left_fisheye` | 960×768 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `head_right_fisheye.mp4` | `observation.images.head_right_fisheye` | 960×768 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `back_left_fisheye.mp4` | `observation.images.back_left_fisheye` | 960×768 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |
| `back_right_fisheye.mp4` | `observation.images.back_right_fisheye` | 960×768 | hevc | yuvj420p | 375 | 30.0 | 12.5000 s |

**fps uniformity (verification A): all 8 streams report avg_rate = 30.0,
guessed_rate = 30.0, base_rate = 30.0 (PyAV `streams.video[0]`). Frame counts
match across all 8 (375 each). Single `info.fps = 30` in LeRobot v3 is safe
for this sample — no resampling required.** Note however the upstream
`script:53` declares `"video.fps": 30.0` *and* `script:196` re-encodes with
`-r 30`, so even if upstream rates ever drift, the official tool forces 30
fps.

The two fisheye groups have different intrinsics (`fx/fy ≈ 475` for
head/right/left fisheyes vs `831` for back fisheyes) per `parameter.json`;
shape is identical so no per-camera resizing needed.

Note `script:51` declares the head-cam shape as `[480, 640, 3]`
(HWC), but `docs/schema-lerobot-v3.md` §2 example shows v3 video features as
`[3, 480, 640]` (CHW) with `names=[channel, height, width]`. This is a
documented LeRobot v3 invariant, not an AgiBot concern, but the converter
must rewrite the order — see `schema-mapping.md` §4.

### 4.2 Depth — `observation/<uuid>/depth/<frame_idx>/{head,hand_left,hand_right}.png`

| Property | Value |
|---|---|
| Per-frame layout | one directory per integer frame index (`0/`, `1/`, …, `374/`) |
| Total depth dirs | 375 (matches HDF5 frame count) |
| PNG channels | 3 cameras: `head.png`, `hand_left.png`, `hand_right.png` (no fisheye depth) |
| Per-PNG mode / dtype | `I;16` / uint16 |
| Per-PNG resolution | 640×480 (matches RGB) |
| Per-PNG range | `[~46, ~1846]` (millimeter integer; max ≈ 1.85 m) |
| Conversion to meters | divide by 1000 (`script:233`) |

Official converter only ingests the `head` channel as
`observation.images.cam_top_depth` (`script:316`), declared `[480, 640, 1]`
`dtype=image` (`script:62-65`). The two `hand_*` depth PNGs are unused by
the upstream converter; out-of-scope for v0.1.

### 4.3 `task_info.json`

Relevant fields for the converter (`task_info.json` is the only NL-task source):

| Field | Used | Value (this sample) |
|---|---|---|
| `task_name` | yes — `task_index` in v3 | `"place_objects_into_handbag"` |
| `task_id` | maybe | `"digitaltwin_3"` |
| `episode_id` | yes — UUID for episode key | `"000aa0b4-..."` |
| `label_info.action_config[].skill` | optional — could become subtasks | `"Grasp"` (0–61), `"Pick"` (62–191), `"Place"` (192–375) |
| `label_info.action_config[].action_text` | empty in this sample | — |
| `init_scene_text` | empty | — |
| `key_frame` | empty | — |

LeRobot v3 has a `meta/subtasks.parquet` slot (see
`docs/schema-lerobot-v3.md` §1) which would naturally hold the `action_config`
slices; deferred for v0.1.

### 4.4 `parameter.json`

Camera intrinsics (`fx, fy, ppx, ppy`, `width`, `height`) and 4×4 extrinsic
pose matrices for all 8 cameras. **LeRobot v3 has no canonical place for
camera intrinsics in the dataset spec** (videos are stored as raw streams; no
`info.cameras` field exists in `DatasetInfo` per
`docs/schema-lerobot-v3.md` §2). For v0.1 we drop these. They could be
preserved in a sidecar `meta/extra/parameter.json` if a downstream tool needs
calibration, but that is out of scope.

---

## 5. Verified findings (verification duties A / B / C)

**A. Per-camera fps uniformity — VERIFIED uniform at 30.0 Hz across all 8
mp4s** (PyAV `average_rate`, `guessed_rate`, `base_rate` all match). LeRobot
v3 single `info.fps` is sufficient. No resampling needed.

**B. Timestamp monotonicity — VERIFIED strictly monotonically increasing.**
Period `0.016666...` s, std `0.0`. **However the implied rate is 60 Hz, not
30.** The HDF5 spans only 6.233 s while each mp4 spans 12.5 s. Both contain
exactly 375 samples. Two equally-likely interpretations:

1. **The `timestamp` column is wrong** (sim-time bug: counter ticked at 60
   Hz instead of 30, but each tick still corresponds to one mp4 frame).
   Treat the HDF5 row index `i` as the canonical frame index and ignore the
   numerical value of `timestamp`. The official LeRobot v3 invariant is
   `timestamp = frame_index / fps` (per `docs/schema-lerobot-v3.md` §5)
   anyway, so the downstream value is recomputed from `frame_index / 30`
   regardless of what AgiBot's column says.

2. **Each h5 row corresponds to a 60 Hz proprio sample, not a 30 Hz video
   frame.** If true, this would mean we have proprio for only the first half
   of the video and the H5 column count of 375 coinciding with the mp4 frame
   count of 375 is sheer coincidence. Implausible — see B-lerobot's earlier
   finding that the official script asserts `len(states_value) ==
   len(depth_imgs)` (`script:318-323`) and depth has exactly 375 frames; a
   true 60 Hz proprio over 6 s would need to be paired against 375 depth
   frames sampled at 60 Hz, which would in turn imply video at 60 Hz — but
   PyAV says 30. Interpretation 1 is the only consistent reading.

**Recommendation:** treat HDF5 row index as ground truth for frame
correspondence; emit `timestamp = frame_index / 30.0` per LeRobot v3 rule;
**do not** propagate the AgiBot `timestamp` column. Document this as a known
upstream issue.

**C. action / state dim match — VERIFIED.**

- `state/joint/position` and `action/joint/position` both `(375, 34)`.
- They are **bitwise identical** in this sample (max abs diff = 0.0). The
  raw `action/joint/position` is therefore *not* a useful action signal on
  its own; it equals state.
- The official script's recipe (`script:293-302`): action[i] := state[i+1] −
  state[i] for i in `[0, N-2]`, then duplicate the last delta to length N.
  This produces an N-length **delta-action** vector aligned to state.
- After 22-joint subselection (§3) the feature dim is **22** for both
  `observation.state` and `action`, matching `script:150-157`.

---

## 6. Bugs / issues in `convert_to_lerobot.py`

| # | Line | Severity | Issue |
|---|---|---|---|
| 1 | 243 | **blocker** | Opens `aligned_joints.h5`. Actual filename is `proprio_states.h5`. README:259 calls it `proprio_stats.h5`. Three different names, none consistent. We use `proprio_states.h5`. |
| 2 | 51,68,80,92,…  | invariant violation | Declares video `shape=[480,640,3]` (HWC). LeRobot v3 stores videos as `[C,H,W]` per `docs/schema-lerobot-v3.md` §2 example. The converter must rewrite the shape/`names` order. |
| 3 | 199 | resolution mismatch | Re-encode pass forces `scale=640:360`. The fisheye sources are 960×768 native and `FEATURES` declares them `[748, 960, 3]`. Output 640×360 ≠ declared shape — would fail v3 reader's tolerance. Skip the upstream re-encode; encode at native resolution. |
| 4 | 92 | minor | Declares fisheye `shape=[748, 960, 3]` but the actual mp4 is **768**×960 (verified). Off-by-20 typo. |
| 5 | 280 | latent | `joint_names.index()` raises `ValueError` if a name is missing. Upstream relies on the 34-name list being stable. We should defensively check and surface a clear error. |
| 6 | 233 | minor | `load_depths` reads only the `head` channel and silently drops `hand_left`/`hand_right` depth. Documented as upstream choice; we follow but note the loss. |
| 7 | 184 | external | Imports `modified_lerobot_dataset.AgiBotDataset` — a wrapper not provided in the sample tarball. Means the upstream script is **non-runnable as released** without the `modified_lerobot_dataset.py` shim, which exists at `data/agibot_sample/scripts/modified_lerobot_dataset.py` (verified). Re-implement from scratch rather than import. |
| 8 | 348 | semantic | Returns `task` as the raw string `task_name`. v3 `tasks.parquet` accepts arbitrary strings, so passing `place_objects_into_handbag` is fine; we may want to substitute spaces for underscores for human readability. Defer to E (Validator). |

---

## 7. Out of scope for v0.1

| Field | Why deferred |
|---|---|
| `state/end/{position,orientation}` | LeRobot v3 has no canonical TCP feature; would conflict with joint state |
| `state/robot/{position,orientation,*_drift}` | Mobile base out of policy scope |
| `action/robot/velocity` | Same; also schema-drift `(375,)` vs README `(375, 2)` (§2.5) |
| `state/joint/{velocity,effort,current_value}` | Effort is a placeholder; velocity is computable from finite-difference |
| `state/effector/{force,index}` | Force is all-zero placeholder; index is bookkeeping |
| `*/end/{angular,velocity,wrench}` | All have shape `(0,)` — empty placeholders |
| Hand depth (`hand_left.png`, `hand_right.png`) | Official script ignores; v0.1 follows |
| `parameter.json` intrinsics/extrinsics | No v3 feature slot |
| `task_info.json.label_info.action_config` | v3 `subtasks.parquet` slot is optional |
| Multi-episode batches | v0.1 is single-episode → single-LeRobot dataset |
| Language-instruction strings | `action_text` is empty in this sample; nothing to attach |

---

## 8. Sources

- `data/agibot_sample/README.md` (lines cited inline as `README:LINE`)
- `data/agibot_sample/scripts/convert_to_lerobot.py` (lines cited inline as `script:LINE`)
- `data/agibot_sample/meta_info/digitaltwin_3/000aa0b4-.../proprio_states.h5` (h5py inspection)
- `data/agibot_sample/observations/digitaltwin_3/000aa0b4-.../video/*.mp4` (PyAV stream metadata)
- `data/agibot_sample/observations/digitaltwin_3/000aa0b4-.../depth/0/*.png` (PIL inspection)
- `docs/schema-lerobot-v3.md` (anchor §2, §5)
