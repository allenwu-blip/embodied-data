from pathlib import Path

from rich.console import Console

console = Console()


def convert_agibot_to_lerobot_v3(*, src: Path, dst: Path) -> None:
    """Convert AgiBot World dataset to LeRobotDataset v3 format.

    Target bugs (from upstream issue tracker):
      - AgiBot-World#18: ValueError in save_episode() / keyword mismatch with LeRobotDataset
      - AgiBot-World#124: KeyError 'actions' column not found
      - AgiBot-World#149: frame/video misalignment after v3 port
      - lerobot#2158: "Directory not empty" and non-monotonic timestamp in v3 write
      - lerobot#2689: joint action spikes after v2.1 -> v3 conversion
    """
    raise NotImplementedError(
        "convert agibot -> lerobot-v3: implementation pending. "
        "Next step: download a small AgiBot World sample and inspect structure."
    )
