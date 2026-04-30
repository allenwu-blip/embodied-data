from __future__ import annotations

import subprocess
from importlib import metadata as _md
from pathlib import Path
from typing import Annotated

import typer

from embodied_data import __version__
from embodied_data._emit import emit_json, get_console
from embodied_data._state import state

app = typer.Typer(
    name="embodied-data",
    help="Cross-format converter and validator for embodied AI datasets.",
    no_args_is_help=True,
    add_completion=False,
)
console = get_console()


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _git_short_sha() -> str:
    """Return short git sha of the package checkout, or 'unknown' when not in a repo."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _build_date() -> str:
    """Return ISO build date: HEAD commit time when in checkout, else dist metadata."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cI", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # Fallback: importlib.metadata may carry a build date in PKG-INFO; otherwise unknown.
    try:
        meta = _md.metadata("embodied-data")
        return str(meta.get("Build-Date") or meta.get("Date") or "unknown")
    except _md.PackageNotFoundError:
        return "unknown"


def _emit_version() -> None:
    sha = _git_short_sha()
    built = _build_date()
    if state.json_output:
        emit_json({"version": __version__, "git_sha": sha, "build_date": built})
    else:
        console.print(f"embodied-data {__version__} (sha={sha}, built={built})")


# ---------------------------------------------------------------------------
# Global --json + --version on the app callback
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of tables."),
    ] = False,
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version, git sha, and build date.",
        ),
    ] = False,
) -> None:
    """Cross-format converter and validator for embodied AI datasets."""
    # Reset state on every invocation (module-global; can leak across tests).
    state.json_output = json_output
    if show_version:
        _emit_version()
        raise typer.Exit()
    # When no subcommand is given and no flag triggered an action, fall through
    # to typer's default no-args-is-help behavior.
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def convert(
    src: Annotated[Path, typer.Argument(help="Source dataset directory.")],
    dst: Annotated[Path, typer.Argument(help="Destination directory.")],
    from_format: Annotated[
        str, typer.Option("--from", help="Source format: agibot|lerobot-v2|lerobot-v3")
    ],
    to_format: Annotated[str, typer.Option("--to", help="Target format: lerobot-v3|agibot")],
    max_episodes: Annotated[
        int | None,
        typer.Option("--max-episodes", help="Cap the number of episodes processed (batch mode)."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Skip episodes already present in dst (batch mode)."),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Parallel episode load workers (default 1)."),
    ] = 1,
) -> None:
    """Convert a dataset from one format to another."""
    from embodied_data.convert import run_convert

    run_convert(
        src=src,
        dst=dst,
        from_format=from_format,
        to_format=to_format,
        max_episodes=max_episodes,
        resume=resume,
        workers=workers,
    )


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


@app.command(name="inspect")
def inspect_cmd(
    path: Annotated[Path, typer.Argument(help="Single .h5 / .hdf5 or .parquet file.")],
) -> None:
    """Dump schema of a single HDF5 or parquet file."""
    from embodied_data.inspect import run_inspect

    run_inspect(path=path)


@app.command()
def version() -> None:
    """Print version, git sha, and build date."""
    _emit_version()


if __name__ == "__main__":
    app()
