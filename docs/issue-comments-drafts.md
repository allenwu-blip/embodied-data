# Upstream issue comment drafts — pending Allen review

These are drafts for posting on the 10 triaged upstream issues. **Nothing is posted until Allen approves [PUBLISH] per issue.** When approved, the Tech Lead will paste the draft via `gh issue comment <repo>#<num> --body "..."` and update the Status line below.

Repository: <https://github.com/allenwu-blip/embodied-data>

---

## In-scope (5) — recommend posting first

### #18 — AgiBot-World `convert_to_lerobot.py` ValueError + visualizer TypeError
- **Target URL**: <https://github.com/OpenDriveLab/AgiBot-World/issues/18>
- **Fix commits**: `75a7d05` (feat(convert)), `dc7e055` (feat(validate))
- **Draft**:
  > Hit the same `convert_to_lerobot.py` ValueError and visualizer breakage. Built `embodied-data` ([repo](https://github.com/allenwu-blip/embodied-data)) which replaces the conversion path (AgiBot World → LeRobot v3) and adds a `validate` command that catches the post-conversion errors before training. Filename mismatch (`aligned_joints.h5` vs actual `proprio_states.h5`) is one of the failure modes we picked up — handled in `embodied-data convert`. If you can share a small slice we'd like to confirm the fix on your data.
- **Status**: pending Allen review

### #124 — `KeyError: Column 'actions' not in dataset` (lerobot v2.1)
- **Target URL**: <https://github.com/OpenDriveLab/AgiBot-World/issues/124>
- **Fix commit**: `75a7d05`
- **Draft**:
  > The nested-vs-flat action-key mismatch is exactly what `embodied-data convert` ([repo](https://github.com/allenwu-blip/embodied-data)) targets — it emits LeRobot v3 with the flat `action` column the current trainer expects (22-dim, head/body/arms/effectors subselection per the official AgiBot recipe). Single-episode v0.0.2 is in main; multi-episode batching is the next sprint. Happy to test on your AgiBot subset if you point at the task IDs.
- **Status**: pending Allen review

### #149 — Frame/video misalignment in 9 tasks during LeRobot 3.0 conversion
- **Target URL**: <https://github.com/OpenDriveLab/AgiBot-World/issues/149>
- **Fix commits**: `75a7d05` (convert), `dc7e055` (validate)
- **Draft**:
  > Frame/video alignment is one of the explicit checks in `embodied-data validate` ([repo](https://github.com/allenwu-blip/embodied-data)). The `convert` command writes a v3 dataset directly without re-entering the upstream multi-episode cache path that surfaces this class of bug. Would like to run it against tasks 362, 543, 392, 532, 361, 570, 475, 595, 620 to confirm we either avoid the misalignment outright or surface it deterministically — let us know if a small slice can be shared.
- **Status**: pending Allen review

### #2158 — Local dataset → LeRobot v3 conversion errors (cache + invalid-timestamp)
- **Target URL**: <https://github.com/huggingface/lerobot/issues/2158>
- **Fix commits**: `75a7d05` (convert), `dc7e055` (validate)
- **Draft**:
  > Both the "non-empty cache dir on episode 2" and the "invalid timestamps on episode 4" symptoms fall in the band `embodied-data validate` ([repo](https://github.com/allenwu-blip/embodied-data)) is designed to surface (timestamp monotonicity is one of the four checks). `embodied-data convert` writes v3 in a single shot rather than re-entering the per-episode cache path, and re-encodes video with `bf=0 g=2` monotonic PTS to avoid the muxer's invalid-DTS rejection. Happy to repro on a small slice.
- **Status**: pending Allen review

### #2689 — ALOHA sim v2.1→v3.0 yields abnormal "spark" joint actions (partial)
- **Target URL**: <https://github.com/huggingface/lerobot/issues/2689>
- **Fix commit**: `dc7e055` (validate, partial)
- **Draft**:
  > ALOHA HDF5 ingest isn't in `embodied-data` v0.1's pair (we do AgiBot↔LeRobot v3 only — [repo](https://github.com/allenwu-blip/embodied-data)), but `embodied-data validate` can quickly tell you whether the converted v3 dataset has action-dim drift or non-monotonic timestamps — useful to isolate "bad data" vs. "bad training run" before chasing it in the trainer. Worth a 30-second `validate` on the converted dataset.
- **Status**: pending Allen review

---

## Out-of-scope (5) — optional polite decline; lower priority

### #132 — Convert ZhiYuan G1 real-robot data to AgiBot-World format
- **Target URL**: <https://github.com/OpenDriveLab/AgiBot-World/issues/132>
- **Draft**:
  > Cross-embodiment ingest from arbitrary collection rigs into AgiBot-World layout isn't in `embodied-data` v0.1 — we currently only do AgiBot↔LeRobot v3 ([repo](https://github.com/allenwu-blip/embodied-data)). Action-space retargeting is on the deferred list; will flag here if/when it lands.
- **Status**: pending Allen review (optional)

### #1329 — Size mismatch [32] vs [6] evaluating fine-tuned VLA
- **Target URL**: <https://github.com/huggingface/lerobot/issues/1329>
- **Draft**:
  > This looks like a model-side normalization shape issue rather than a dataset format problem, so it's outside `embodied-data` v0.1's scope ([repo](https://github.com/allenwu-blip/embodied-data)). Flagging only because a few similar reports get bucketed under "conversion" upstream.
- **Status**: pending Allen review (optional)

### #2446 — Better way to manage dataset versions on the Hub (partial)
- **Target URL**: <https://github.com/huggingface/lerobot/issues/2446>
- **Draft**:
  > Hub PR-workflow piece is outside `embodied-data` ([repo](https://github.com/allenwu-blip/embodied-data)) — we're a CLI, not a hub UX. For the conversion subproblem we currently only cover AgiBot↔LeRobot v3; a generic v2.0→v3.0 LeRobot upgrade is on the deferred list.
- **Status**: pending Allen review (optional)

### #2679 — `merge_datasets` strips fps from scalar features
- **Target URL**: <https://github.com/huggingface/lerobot/issues/2679>
- **Draft**:
  > Internal `merge_datasets` bug rather than a format-conversion issue, so it's outside `embodied-data` v0.1's scope ([repo](https://github.com/allenwu-blip/embodied-data)). `embodied-data validate` does flag missing fps on scalar features, which can at least catch the broken state before it propagates downstream.
- **Status**: pending Allen review (optional)

### #336 — rerun always plays at 30 fps regardless of dataset fps
- **Target URL**: <https://github.com/huggingface/lerobot/issues/336>
- **Draft**:
  > `embodied-data` v0.1 doesn't ship a visualizer ([repo](https://github.com/allenwu-blip/embodied-data)) — `preview` only prints stats, including the dataset's declared fps. So we can confirm the dataset side is correct but can't fix rerun's playback default.
- **Status**: pending Allen review (optional)

---

## Posting protocol (when approved)

1. Allen replies "[PUBLISH] approve #X" listing approved issue numbers.
2. Tech Lead posts via `gh issue comment <owner>/<repo>#<num> --body-file <(echo "...")` for each.
3. Update **Status** line to `posted YYYY-MM-DD by allenwu-blip, comment URL <...>`.
4. Never edit a draft after posting — track replies separately.
