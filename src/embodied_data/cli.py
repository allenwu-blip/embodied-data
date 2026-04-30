from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from embodied_data import __version__

app = typer.Typer(
    name="embodied-data",
    help="Cross-format converter and validator for embodied AI datasets.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def convert(
    src: Annotated[Path, typer.Argument(help="Source dataset directory.")],
    dst: Annotated[Path, typer.Argument(help="Destination directory.")],
    from_format: Annotated[
        str, typer.Option("--from", help="Source format: agibot|lerobot-v2|lerobot-v3")
    ],
    to_format: Annotated[str, typer.Option("--to", help="Target format: lerobot-v3|agibot")],
) -> None:
    """Convert a dataset from one format to another."""
    from embodied_data.convert import run_convert

    run_convert(src=src, dst=dst, from_format=from_format, to_format=to_format)


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Dataset directory to validate.")],
    fmt: Annotated[str, typer.Option("--format", help="Format (default: auto-detect).")] = "auto",
) -> None:
    """Check fps / timestamps / action dim / frame alignment."""
    from embodied_data.validate import run_validate

    run_validate(path=path, fmt=fmt)


@app.command()
def preview(
    path: Annotated[Path, typer.Argument(help="Dataset directory to preview.")],
    n: Annotated[int, typer.Option("-n", help="Number of episodes to sample.")] = 10,
) -> None:
    """Print stats for the first N episodes."""
    from embodied_data.preview import run_preview

    run_preview(path=path, n=n)


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"embodied-data {__version__}")


if __name__ == "__main__":
    app()
