from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.table import Table

from embodied_data._emit import emit_json

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


_STATUS_STYLE = {
    "PASS": "green",
    "WARN": "yellow",
    "FAIL": "red",
    "SKIP": "dim",
}


def render(
    console: Console,
    *,
    path: str,
    fmt: str,
    results: list[CheckResult],
) -> None:
    console.print(f"Validation: {path}  (format: {fmt})")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")
    for r in results:
        table.add_row(r.name, f"[{_STATUS_STYLE[r.status]}]{r.status}[/]", r.detail)
    console.print(table)

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    if n_fail:
        console.print(f"Result: [red]FAIL[/red]  ({n_fail} issue{'s' if n_fail != 1 else ''})")
    elif n_warn:
        console.print(f"Result: [yellow]PASS with {n_warn} warning(s)[/yellow]")
    else:
        console.print("Result: [green]PASS[/green]")


def render_json(
    *,
    path: str,
    fmt: str,
    results: list[CheckResult],
    exit_code_value: int,
) -> None:
    overall = "PASS"
    if any(r.status == "FAIL" for r in results):
        overall = "FAIL"
    elif any(r.status == "WARN" for r in results):
        overall = "WARN"
    payload = {
        "path": path,
        "format": fmt,
        "results": [{"name": r.name, "status": r.status, "detail": r.detail} for r in results],
        "result": overall,
        "exit_code": exit_code_value,
    }
    emit_json(payload)


def exit_code(results: list[CheckResult]) -> int:
    if any(r.status == "FAIL" for r in results):
        return 1
    return 0


def suggestion_for(result: CheckResult, *, dataset_path: str) -> str | None:
    """Return a next-step hint for a failing check, or None if generic."""
    name = result.name
    if name == "timestamp monotonicity":
        return (
            f"run `embodied-data inspect {dataset_path}/proprio_states.h5` to see "
            "the offending frame range"
        )
    if name == "frame-video alignment":
        return (
            "h5 row count diverges from video frame count — re-export the episode "
            "or trim the longer side; see issue #149"
        )
    if name == "action-dim consistency":
        return (
            "state and action arrays have mismatched shapes; verify the upstream "
            "AgiBot HDF5 contains both /state/joint/position and /action/joint/position "
            "with identical (frames, dim)"
        )
    if name == "fps consistency":
        return "different mp4s report different fps; re-encode the inconsistent ones to 30fps"
    return None
