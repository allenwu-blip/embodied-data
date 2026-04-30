"""Process-wide CLI state.

Kept in a separate module so submodules (validate/, preview/, convert/) can
import it without depending on cli.py and triggering circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlobalState:
    json_output: bool = False


state = GlobalState()
