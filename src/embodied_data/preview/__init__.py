from pathlib import Path

from rich.console import Console

console = Console()


def run_preview(*, path: Path, n: int) -> None:
    """Print stats for the first N episodes: fps distribution, action range, camera count."""
    raise NotImplementedError(
        "preview: implementation pending. "
        "Report shape: episode count, fps histogram, action dim + range, camera names, total hours."
    )
