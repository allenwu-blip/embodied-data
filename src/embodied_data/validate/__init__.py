from pathlib import Path

from embodied_data._emit import emit_error, emit_json, get_console
from embodied_data._state import state
from embodied_data.validate import agibot, lerobot_v3
from embodied_data.validate._report import (
    CheckResult,
    exit_code,
    render,
    render_json,
    suggestion_for,
)

console = get_console()


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
        emit_error(
            f"path does not exist: {path}",
            suggestion="pass an existing dataset directory",
            exit_code=2,
        )

    if fmt == "auto":
        fmt = _detect(path)

    results: list[CheckResult]
    if fmt == "lerobot-v3":
        results = lerobot_v3.validate(path)
    elif fmt == "agibot":
        results = agibot.validate(path)
    else:
        emit_error(
            f"unknown format for {path}",
            suggestion=(
                "use `--format lerobot-v3` or `--format agibot` to override; "
                "expected meta/info.json or proprio_states.h5 in path"
            ),
            exit_code=2,
        )
        return  # unreachable; emit_error raises

    code = exit_code(results)
    if state.json_output:
        render_json(path=str(path), fmt=fmt, results=results, exit_code_value=code)
    else:
        render(console, path=str(path), fmt=fmt, results=results)
        # Surface per-check suggestions for FAILs in non-JSON mode.
        for r in results:
            if r.status == "FAIL":
                hint = suggestion_for(r, dataset_path=str(path))
                if hint:
                    console.print(f"[dim]suggestion:[/dim] {hint}")
    if code != 0:
        raise SystemExit(code)


__all__ = ["run_validate", "emit_json"]
