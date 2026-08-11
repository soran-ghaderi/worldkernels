r"""Cached rich Console plus terminal capability detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_CONSOLE: Any = None


@dataclass(frozen=True)
class Capabilities:
    is_terminal: bool
    no_color: bool
    width: int
    is_ci: bool


def console() -> Any:
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console

        _CONSOLE = Console()
    return _CONSOLE


def capabilities() -> Capabilities:
    con = console()
    return Capabilities(
        is_terminal=bool(con.is_terminal),
        no_color=bool(os.environ.get("NO_COLOR")),
        width=con.width,
        is_ci=bool(os.environ.get("CI")),
    )


def is_quiet_env() -> bool:
    return bool(os.environ.get("WORLDKERNELS_QUIET"))
