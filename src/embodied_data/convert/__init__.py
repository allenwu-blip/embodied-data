from pathlib import Path

from rich.console import Console

console = Console()

_SUPPORTED = {
    ("agibot", "lerobot-v3"),
}


def run_convert(*, src: Path, dst: Path, from_format: str, to_format: str) -> None:
    pair = (from_format, to_format)
    if pair not in _SUPPORTED:
        console.print(
            f"[red]Unsupported conversion: {from_format} -> {to_format}[/red]\n"
            f"Supported pairs in v0.0.1: {sorted(_SUPPORTED)}"
        )
        raise SystemExit(2)

    if pair == ("agibot", "lerobot-v3"):
        from embodied_data.convert.agibot_to_lerobot import convert_agibot_to_lerobot_v3

        convert_agibot_to_lerobot_v3(src=src, dst=dst)
