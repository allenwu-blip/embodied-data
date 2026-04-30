# Upstream issues covered by embodied-data v0.1

| # | Issue | One-line problem | v0.1 fix |
|---|-------|------------------|----------|
| 1 | [AgiBot-World#18](https://github.com/OpenDriveLab/AgiBot-World/issues/18) | Conversion script throws `ValueError`; visualizer throws `TypeError` on unexpected kwarg. | `convert` + `validate` |
| 2 | [AgiBot-World#124](https://github.com/OpenDriveLab/AgiBot-World/issues/124) | After AgiBot→LeRobot v2.1 conversion, training fails with `KeyError: 'actions'` because only nested `actions.effector.position` exists. | `convert` |
| 3 | [AgiBot-World#132](https://github.com/OpenDriveLab/AgiBot-World/issues/132) | User wants to convert their own ZhiYuan G1 wheeled dual-arm data into AgiBot-World format to fine-tune GO-1. | out of scope (cross-embodiment retargeting / non-AgiBot raw inputs) |
| 4 | [AgiBot-World#149](https://github.com/OpenDriveLab/AgiBot-World/issues/149) | Official AgiBot→LeRobot v3 script produces frame/video misalignment on 9 specific tasks, breaking dataset load. | `convert` + `validate` |
| 5 | [lerobot#2158](https://github.com/huggingface/lerobot/issues/2158) | Local-dataset → LeRobot v3 conversion fails: `OSError` non-empty cache dir on episode 2, then video concat invalid-timestamp error on episode 4. | `convert` + `validate` (timestamp monotonicity) |
| 6 | [lerobot#2689](https://github.com/huggingface/lerobot/issues/2689) | After ALOHA sim insertion v2.1→v3.0 conversion, training shows abnormal "spark" joint actions. | `validate` (action-dim/consistency check) — partial; root cause may be upstream |
| 7 | [lerobot#1329](https://github.com/huggingface/lerobot/issues/1329) | Fine-tuned SmolVLA/ACT eval fails with size mismatch: saved norm buffers shape [32] vs expected [6]. | out of scope (model-side normalization, not dataset format) |
| 8 | [lerobot#2446](https://github.com/huggingface/lerobot/issues/2446) | Hub datasets fragmented across v2.0/v2.1/v3.0; community can't update without write access. | out of scope (hub workflow / PR mechanism) — `convert` partially helps the v2→v3 step |
| 9 | [lerobot#2679](https://github.com/huggingface/lerobot/issues/2679) | `merge_datasets` strips `fps` from scalar features (timestamp, frame_index, episode_index), breaking later adds. | out of scope (LeRobot internal merge bug, not a format conversion) |
| 10 | [lerobot#336](https://github.com/huggingface/lerobot/issues/336) | rerun visualizer always plays at 30 fps regardless of dataset fps. | out of scope (visualizer playback bug) — `preview` reports correct fps, doesn't fix rerun |

**Tally**: 5 in-scope, 5 out-of-scope (one of which — #8 — is partially addressable).

## Detailed notes

### [AgiBot-World#18](https://github.com/OpenDriveLab/AgiBot-World/issues/18) — Data convert and visualize problem
**Problem**: User hits a `ValueError` when running the AgiBot data conversion script, and a `TypeError` (unexpected keyword argument) when running the visualizer.
**v0.1 coverage**: In scope. `convert` replaces the upstream conversion path (AgiBot World ↔ LeRobot v3) and `validate` catches the post-conversion errors that the visualizer would otherwise hit at load time.
**Comment draft (for approval before posting)**: We hit the same conversion + visualizer breakage and built `embodied-data convert` plus `embodied-data validate` to handle it end-to-end. If you'd like to try it on your data we'd appreciate the repro.

### [AgiBot-World#124](https://github.com/OpenDriveLab/AgiBot-World/issues/124) — KeyError: Column 'actions' not in dataset (lerobot v2.1)
**Problem**: User converted AgiBot data to LeRobot v2.1 with an older `any4lerobot`, but training fails because the dataset only has nested `actions.effector.position` columns — no flat `actions` key.
**v0.1 coverage**: In scope. `convert` produces a LeRobot v3 dataset with action keys laid out per the current LeRobot schema, eliminating the nested-vs-flat mismatch.
**Comment draft (for approval before posting)**: This nested-action layout mismatch is exactly what `embodied-data convert` targets — it emits LeRobot v3 with the action columns the current trainer expects. Happy to test against your AgiBot subset if you can share the task IDs.

### [AgiBot-World#132](https://github.com/OpenDriveLab/AgiBot-World/issues/132) — Convert ZhiYuan G1 real-robot data to AgiBot-World format
**Problem**: User collected data on a ZhiYuan Genie G1 wheeled dual-arm and wants to reshape it into AgiBot-World format to fine-tune GO-1.
**v0.1 coverage**: Out of scope. v0.1 only handles the AgiBot World ↔ LeRobot v3 pair; ingesting an arbitrary proprietary collection format and remapping action spaces is cross-embodiment retargeting, which is explicitly deferred.
**Comment draft (for approval before posting)**: Cross-embodiment ingest from arbitrary collection rigs into AgiBot-World layout isn't something `embodied-data` v0.1 covers — we only do AgiBot↔LeRobot v3. We'll flag this if/when retargeting lands.

### [AgiBot-World#149](https://github.com/OpenDriveLab/AgiBot-World/issues/149) — Frame/video misalignment in 9 tasks during LeRobot 3.0 conversion
**Problem**: After running the official AgiBot→LeRobot 3.0 conversion script, 9 tasks (362, 543, 392, 532, 361, 570, 475, 595, 620) have data frames misaligned with video frames, causing load errors.
**v0.1 coverage**: In scope. `convert` is the replacement pipeline; `validate` includes a frame-video alignment check that surfaces this class of bug before it reaches training.
**Comment draft (for approval before posting)**: Frame/video alignment is one of the explicit checks in `embodied-data validate`, and `convert` is meant to replace the upstream script. We'd like to run it against these 9 tasks to confirm we catch (and ideally fix) the misalignment.

### [lerobot#2158](https://github.com/huggingface/lerobot/issues/2158) — Local dataset → LeRobot v3 conversion errors
**Problem**: Conversion that worked under v2.1 fails on v3: episode 2 hits `OSError` "directory not empty", and episode 4 hits a video-concat "invalid timestamps" error. User patched out the cache cleanup to bypass the first, but the timestamp error persists.
**v0.1 coverage**: In scope. `convert` produces v3 directly without the multi-episode cache-stomp pattern, and `validate` includes timestamp monotonicity to catch the v4-episode failure mode at the dataset level.
**Comment draft (for approval before posting)**: The "non-empty cache dir on episode 2" and "invalid timestamps on episode 4" both fall in the band `embodied-data validate` is designed to surface, and `convert` writes v3 in one shot rather than re-entering the per-episode cache path. Happy to repro on a small slice if you can share one.

### [lerobot#2689](https://github.com/huggingface/lerobot/issues/2689) — ALOHA sim v2.1→v3.0 yields abnormal "spark" joint actions
**Problem**: After the official v2.1→v3.0 conversion of `aloha_sim_insertion`, training exhibits unusual joint motion ("spark of joint actions") in simulation.
**v0.1 coverage**: Partially in scope. v0.1 doesn't touch ALOHA HDF5 ingest, but `validate` (action-dim consistency, timestamp monotonicity) can flag whether the converted v3 dataset is the source of the anomaly vs. a downstream training-side issue. The conversion itself is out of scope (different format pair).
**Comment draft (for approval before posting)**: ALOHA HDF5 ingest isn't in `embodied-data` v0.1's pair (we do AgiBot↔LeRobot v3), but `validate` can quickly tell you whether the converted v3 dataset has action-dim or timestamp drift — useful to isolate "bad data" vs. "bad training run". We'd suggest running it on the converted dataset before chasing it in the trainer.

### [lerobot#1329](https://github.com/huggingface/lerobot/issues/1329) — Size mismatch [32] vs [6] evaluating fine-tuned VLA
**Problem**: Saved SmolVLA/ACT checkpoints have normalization buffers of shape [32], but the eval code expects [6] for the user's 6-DoF arm. Closed as not planned.
**v0.1 coverage**: Out of scope. This is a model-side normalization-buffer-shape bug, not a dataset format conversion or a validation issue against the dataset.
**Comment draft (for approval before posting)**: This looks like a model-side normalization shape issue rather than a dataset format problem, so it's outside what `embodied-data` v0.1 addresses — flagging here only because a few similar reports are bucketed under "conversion" upstream.

### [lerobot#2446](https://github.com/huggingface/lerobot/issues/2446) — Better way to manage dataset versions on the Hub
**Problem**: Hub has official LeRobot datasets stranded at v2.0 while the codebase is on v3.0. Community can't bring them current without write access; OP suggests a v2.0→v3.0 script and a `push_to_hub`-as-PR flow.
**v0.1 coverage**: Mostly out of scope (hub permissions / PR workflow is product-side). The v2→v3 conversion piece is partially addressable by `convert` for the AgiBot pair only; the general v2.0→v3.0 LeRobot-to-LeRobot upgrade is not v0.1 scope.
**Comment draft (for approval before posting)**: The hub PR-workflow piece is outside `embodied-data` v0.1 — we're a CLI, not a hub UX. For the conversion subproblem we currently only cover AgiBot↔LeRobot v3; a generic v2→v3 LeRobot upgrade is on the deferred list.

### [lerobot#2679](https://github.com/huggingface/lerobot/issues/2679) — `merge_datasets` strips fps from scalar features
**Problem**: Calling `merge_datasets` removes the `fps` attribute from scalar columns (timestamp, frame_index, episode_index) while preserving it elsewhere, breaking subsequent merges and episode deletions.
**v0.1 coverage**: Out of scope. This is a bug inside LeRobot's merge function, not a conversion or validation surface. `validate` would detect a missing fps after the fact, but v0.1 does not modify `merge_datasets` behavior.
**Comment draft (for approval before posting)**: This is an internal `merge_datasets` bug rather than a format-conversion issue, so it's outside `embodied-data` v0.1's scope. Our `validate` does flag missing fps on scalar features, which can at least catch the broken state before it propagates.

### [lerobot#336](https://github.com/huggingface/lerobot/issues/336) — rerun always plays at 30 fps regardless of dataset fps
**Problem**: The rerun-based visualizer hardcodes 30 fps playback even when the dataset declares 15 fps or 10 fps. Closed as not planned.
**v0.1 coverage**: Out of scope. v0.1 has no visualization component. `preview` reports the correct fps from the dataset metadata but does not patch rerun's playback.
**Comment draft (for approval before posting)**: `embodied-data` v0.1 doesn't ship a visualizer — `preview` only prints stats, including the dataset's declared fps. So we can confirm the dataset side is correct but can't fix rerun's playback default.
