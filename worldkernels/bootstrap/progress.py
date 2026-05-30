r"""Unified progress UI for the bootstrap pipeline.

Modes:
- ``tty``: rich live panel with per-phase status, glyphs, and bar
- ``plain``: one log line per phase transition
- ``quiet``: silent except for errors
- ``json``: structured one-event-per-line stdout (used by SSE)
"""

from __future__ import annotations

import json as _json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

PhaseStatus = Literal["pending", "running", "done", "skipped", "failed"]

PHASES: tuple[str, ...] = ("resolve", "deps", "packages", "weights", "init")
_LABELS: dict[str, str] = {
    "resolve": "resolving",
    "deps": "runtime deps",
    "packages": "model package",
    "weights": "weights",
    "init": "initializing",
}


@dataclass
class PhaseState:
    name: str
    status: PhaseStatus = "pending"
    message: str = ""
    fraction: float | None = None


class ProgressController:
    r"""Phase-keyed progress UI shared by CLI, HTTP, and Python entry points."""

    def __init__(
        self,
        mode: str = "auto",
        phases: tuple[str, ...] = PHASES,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.mode = _resolve_mode(mode)
        self.phases = {p: PhaseState(p) for p in phases}
        self._live: Any = None
        self._console: Any = None
        self._sink = sink

    def __enter__(self) -> "ProgressController":
        if self.mode == "tty":
            try:
                from rich.console import Console
                from rich.live import Live

                self._console = Console()
                self._live = Live(
                    self._render(),
                    console=self._console,
                    refresh_per_second=10,
                    transient=False,
                )
                self._live.__enter__()
            except ImportError:
                self.mode = "plain"
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.update(self._render(), refresh=True)
            self._live.__exit__(exc_type, exc, tb)

    def event(
        self,
        phase: str,
        status: PhaseStatus,
        message: str = "",
        fraction: float | None = None,
    ) -> None:
        if phase not in self.phases:
            return
        ps = self.phases[phase]
        ps.status = status
        ps.message = message
        ps.fraction = fraction

        if self._sink is not None:
            self._sink({"phase": phase, "status": status, "message": message, "fraction": fraction})

        if self.mode == "tty" and self._live is not None:
            self._live.update(self._render())
        elif self.mode == "plain":
            glyph = {
                "running": "▸",
                "done": "✓",
                "skipped": "·",
                "failed": "✗",
            }.get(status)
            if glyph is not None:
                print(f"{glyph} {_LABELS.get(phase, phase):<15s} {message}", flush=True)
        elif self.mode == "json":
            print(
                _json.dumps(
                    {
                        "phase": phase,
                        "status": status,
                        "message": message,
                        "fraction": fraction,
                    }
                ),
                flush=True,
            )
        # mode == "sink" or "quiet": sink-only (or silent)

    def finalize(self, success: bool, summary: str = "") -> None:
        if self.mode == "tty" and self._console is not None:
            from rich.text import Text

            mark = (
                Text("✓ ready", style="bold green")
                if success
                else Text("✗ failed", style="bold red")
            )
            if summary:
                self._console.print(mark, Text(f"  {summary}", style="dim"))
            else:
                self._console.print(mark)
        elif self.mode == "plain":
            print(("✓ ready " if success else "✗ failed ") + summary, flush=True)
        elif self.mode == "json":
            event = {"phase": "ready" if success else "failed", "message": summary}
            print(_json.dumps(event), flush=True)

    def _render(self):
        from rich.table import Table
        from rich.text import Text

        t = Table(show_header=False, show_edge=False, box=None, padding=(0, 1, 0, 0))
        t.add_column(width=2, justify="center")
        t.add_column(width=15)
        t.add_column()
        for name, ps in self.phases.items():
            glyph = _glyph(ps.status)
            label = Text(_LABELS.get(name, name), style="bold" if ps.status == "running" else "dim")
            body = ps.message
            if ps.fraction is not None:
                bar = _bar(ps.fraction)
                body = f"{ps.message}  {bar} {int(ps.fraction * 100):3d}%"
            t.add_row(glyph, label, Text(body))
        return t


def _resolve_mode(mode: str) -> str:
    if os.environ.get("WORLDKERNELS_QUIET"):
        return "quiet"
    if mode == "auto":
        return "plain"
    return mode


def _glyph(status: PhaseStatus):
    from rich.text import Text

    return {
        "pending": Text("·", style="dim"),
        "running": Text("▸", style="cyan bold"),
        "done": Text("✓", style="green"),
        "skipped": Text("·", style="dim"),
        "failed": Text("✗", style="red bold"),
    }.get(status, Text("·", style="dim"))


def _bar(frac: float, width: int = 16) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)
