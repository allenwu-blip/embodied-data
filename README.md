# embodied-data

Cross-format converter and validator for embodied AI datasets.

## Why this exists

Robotics researchers spend days rewriting the same dataset conversion scripts. AgiBot World's official `convert_to_lerobot.py` has had [unresolved issues](https://github.com/OpenDriveLab/AgiBot-World/issues) for months; LeRobot's v2.0 / v2.1 / v3.0 versions [break each other](https://github.com/huggingface/lerobot/issues/2158); every lab writes its own timestamp alignment check. This tool is the layer that stops.

## v0.0.1 scope (what this does today)

Three commands, one format pair:

```
embodied-data convert   agibot → lerobot-v3
embodied-data validate  fps / timestamp / action-dim / frame alignment
embodied-data preview   stats report for first N episodes
```

Non-goals for v0.0.1: action-space retargeting across embodiments, Chinese prompt embedding, RLDS support. Those come later.

## Install

```bash
uv sync
uv run embodied-data --help
```
