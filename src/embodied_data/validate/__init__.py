from pathlib import Path

from rich.console import Console

console = Console()


def run_validate(*, path: Path, fmt: str) -> None:
    """Check fps / timestamps / action dim / frame alignment.

    Target bugs:
      - lerobot#2679: merging loses fps metadata
      - lerobot#336: rerun viewer forces 30fps regardless of metadata
      - lerobot#2689: non-monotonic timestamps after conversion
      - AgiBot-World#149: frame/video misalignment
    """
    raise NotImplementedError(
        "validate: implementation pending. "
        "Next step: define the invariants (fps consistency, monotonic timestamp, "
        "action-dim uniform across episodes, joint-vs-video frame alignment)."
    )
