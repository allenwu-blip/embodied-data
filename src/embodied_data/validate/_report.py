from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.table import Table

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


def exit_code(results: list[CheckResult]) -> int:
    if any(r.status == "FAIL" for r in results):
        return 1
    return 0
