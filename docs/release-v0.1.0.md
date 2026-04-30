# embodied-data v0.1.0 — AgiBot ↔ LeRobot v3 converter

First public release. A small, opinionated CLI that converts AgiBot World
episodes into LeRobot v3 datasets and validates the result against the
schema footguns we found while reading both projects' source.

## What's in

Four commands:

- `convert` — AgiBot World → LeRobot v3, single episode. 22-dim joint subselect (head + body + dual arms + effectors), first-difference action, `frame_index/30` timestamps, single head camera re-encoded to h264 with B-frames disabled (LeRobot v3 footgun F1).
- `validate` — four checks against a LeRobot v3 directory: fps consistency, timestamp monotonicity, action-dim agreement between `info.json` and the parquet `action` column, and frame-count vs. video-duration alignment. Surfaces the AgiBot upstream 60Hz timestamp bug as a `WARN`.
- `preview` — Rich-formatted stats table for either format (frame total, fps, state/action dims, camera list).
- `inspect` — quick structural dump of an episode (HDF5 tree for AgiBot, parquet column list for LeRobot).

Validated against the open-license `AgiBotDigitalWorld` HF dataset (one paired episode shipped at `data/agibot_sample/`, 124 MB). End-to-end: 375 frames, 1 task, 1 camera (`observation.images.top_head`), `validate` PASS.

## Install

```bash
pip install embodied-data        # once published to PyPI
uv pip install embodied-data     # uv users
```

Python ≥ 3.12 required. `ffmpeg` must be on `PATH` for video re-encoding.

## Quickstart

```bash
embodied-data convert \
    data/agibot_sample/observations/digitaltwin_3/000aa0b4-8fbe-432a-b6ae-559a7d7b3b96 \
    /tmp/out --from agibot --to lerobot-v3
embodied-data validate /tmp/out
embodied-data preview  /tmp/out
```

## Coverage

5 in-scope upstream issues addressed by this release. Full table with per-issue mapping in [`docs/issues-coverage.md`](issues-coverage.md):

- `AgiBot-World#18`, `#124`, `#149` — conversion + validation
- `lerobot#2158` — local v3 conversion (cache + invalid-timestamp)
- `lerobot#2689` — partial; ALOHA conversion itself is out of scope, but `validate` flags action-dim/timestamp drift on the converted output

5 explicitly out-of-scope (cross-embodiment retargeting, hub PR workflow, model-side normalization, internal `merge_datasets` bug, rerun playback).

## Caveats

- Single-episode conversion only. Multi-episode batching with chunk rollover is a deferred sprint.
- Forward path (`agibot → lerobot-v3`) only. Reverse (`lerobot-v3 → agibot`) is deferred.
- Head camera only. The 7 fisheye/hand cameras and the depth stream (uint16 mm PNG) are deferred — schema is documented, code is not.
- The frame/video alignment check uses an episode-metadata proxy (length/fps × duration); per-frame PTS drift inside a video file would not be caught. Acceptable for v0.1; see `WORKLOG.md`.

## Acknowledgments

Built on top of the schemas published by the HuggingFace LeRobot team and the OpenDriveLab AgiBot team. Bugs in this code are mine; the data formats they shipped are the only reason a one-week converter is even possible.
