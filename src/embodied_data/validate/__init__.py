from pathlib import Path

from rich.console import Console

from embodied_data.validate import agibot, lerobot_v3
from embodied_data.validate._report import CheckResult, exit_code, render

console = Console()


def _detect(path: Path) -> str:
    if (path / "meta" / "info.json").exists():
        return "lerobot-v3"
    if (path / "proprio_states.h5").exists():
        return "agibot"
    if list(path.glob("**/proprio_states.h5")):
        return "agibot"
    if (path.parent / "scripts" / "convert_to_lerobot.py").exists():
        return "agibot"
    return "unknown"


def run_validate(*, path: Path, fmt: str) -> None:
    """Check fps / timestamps / action dim / frame alignment."""
    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        raise SystemExit(2)

    if fmt == "auto":
        fmt = _detect(path)

    results: list[CheckResult]
    if fmt == "lerobot-v3":
        results = lerobot_v3.validate(path)
    elif fmt == "agibot":
        results = agibot.validate(path)
    else:
        console.print(
            f"[red]Unknown format for {path}.[/red] "
            "Expected meta/info.json (lerobot-v3) or proprio_states.h5 (agibot). "
            "Pass --format explicitly to override."
        )
        raise SystemExit(2)

    render(console, path=str(path), fmt=fmt, results=results)
    code = exit_code(results)
    if code != 0:
        raise SystemExit(code)
