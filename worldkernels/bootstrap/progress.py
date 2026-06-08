r"""Bootstrap progress UI: thin adapter over worldkernels.ui (tty/plain/quiet/json/sink)."""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any, Callable

from worldkernels.ui.events import PhaseStatus
from worldkernels.ui.progress import PHASES, PhaseProgress

__all__ = ["ProgressController", "PHASES", "PhaseStatus", "cli_mode"]

log = logging.getLogger(__name__)


def cli_mode() -> str:
    r"""Map the resolved CLI output mode to a ProgressController mode string."""
    from worldkernels.ui.verbosity import OutputMode, resolve_output_mode

    return {
        OutputMode.AUTO: "tty",
        OutputMode.PLAIN: "plain",
        OutputMode.JSON: "json",
        OutputMode.QUIET: "quiet",
    }[resolve_output_mode()]


class ProgressController:
    r"""Phase-keyed progress UI shared by CLI, HTTP, and Python entry points."""

    def __init__(
        self,
        mode: str = "auto",
        phases: tuple[str, ...] = PHASES,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.mode = _resolve_mode(mode)
        self._pp = PhaseProgress(phases)
        self._live: Any = None
        self._console: Any = None
        self._sink = sink

    def __enter__(self) -> "ProgressController":
        if self.mode == "tty":
            from worldkernels.ui.console import console
            from worldkernels.ui.live import LiveDisplay
            from worldkernels.ui.verbosity import OutputMode

            self._console = console()
            self._live = LiveDisplay(self._pp.render, OutputMode.AUTO).__enter__()
            if not self._live.active:
                self.mode = "plain"
                self._live = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)

    def event(
        self,
        phase: str,
        status: PhaseStatus,
        message: str = "",
        fraction: float | None = None,
    ) -> None:
        if not self._pp.set(phase, status, message, fraction):
            return

        if self._sink is not None:
            self._sink({"phase": phase, "status": status, "message": message, "fraction": fraction})

        if self.mode == "tty" and self._live is not None:
            self._live.update()
        elif self.mode == "plain":
            ln = self._pp.plain_line(phase, status, message)
            if ln is not None:
                print(ln, flush=True)
        elif self.mode == "json":
            print(
                _json.dumps(
                    {"phase": phase, "status": status, "message": message, "fraction": fraction}
                ),
                flush=True,
            )

    def finalize(self, success: bool, summary: str = "") -> None:
        if self.mode == "tty" and self._console is not None:
            from rich.text import Text

            from worldkernels.ui.theme import style

            mark = (
                Text("✓ ready", style=f"bold {style('success')}")
                if success
                else Text("✗ failed", style=f"bold {style('error')}")
            )
            if summary:
                self._console.print(mark, Text(f"  {summary}", style=style("muted")))
            else:
                self._console.print(mark)
        elif self.mode == "plain":
            print(("✓ ready " if success else "✗ failed ") + summary, flush=True)
        elif self.mode == "json":
            event = {"phase": "ready" if success else "failed", "message": summary}
            print(_json.dumps(event), flush=True)


def _resolve_mode(mode: str) -> str:
    if os.environ.get("WORLDKERNELS_QUIET"):
        return "quiet"
    return "plain" if mode == "auto" else mode
