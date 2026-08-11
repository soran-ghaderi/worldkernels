r"""Phase-keyed and step-keyed progress renderers, gradient bars, sparklines, working status."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from worldkernels.ui.events import PhaseStatus
from worldkernels.ui.verbosity import OutputMode

PHASES: tuple[str, ...] = ("resolve", "deps", "packages", "weights", "init")
_LABELS: dict[str, str] = {
    "resolve": "resolving",
    "deps": "runtime deps",
    "packages": "model package",
    "weights": "weights",
    "init": "initializing",
}

_SPARK = "▁▂▃▄▅▆▇█"
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_VERBS = ("Working", "Crunching", "Churning", "Cooking")


@dataclass
class PhaseState:
    name: str
    status: PhaseStatus = "pending"
    message: str = ""
    fraction: float | None = None


def bar(frac: float, width: int = 16) -> str:
    r"""Glyph progress bar."""
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ramp_at(stops: tuple[str, str, str], t: float) -> str:
    r"""Interpolate a 3-stop hex ramp at ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        a, b, tt = stops[0], stops[1], t * 2
    else:
        a, b, tt = stops[1], stops[2], (t - 0.5) * 2
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    r = round(ar + (br - ar) * tt)
    g = round(ag + (bg - ag) * tt)
    bl = round(ab + (bb - ab) * tt)
    return f"#{r:02X}{g:02X}{bl:02X}"


def sparkline(values: Sequence[float], width: int = 0) -> Any:
    r"""Ramp-tinted ``▁▂▃▅▇`` sparkline of the last ``width`` values (rich Text)."""
    from rich.text import Text

    from worldkernels.ui.theme import ramp

    vals = list(values)
    if width and len(vals) > width:
        vals = vals[-width:]
    t = Text()
    if not vals:
        return t
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    stops = ramp()
    n = len(vals)
    for i, v in enumerate(vals):
        level = int((v - lo) / span * (len(_SPARK) - 1))
        t.append(_SPARK[level], style=_ramp_at(stops, i / max(1, n - 1)))
    return t


def _glyph_text(status: PhaseStatus) -> Any:
    from rich.text import Text

    from worldkernels.ui.theme import GLYPHS, style

    return Text(GLYPHS.get(status, "·"), style=style(status))


class PhaseProgress:
    r"""Five-phase model-load renderer holding phase state and the rich Table."""

    def __init__(self, phases: tuple[str, ...] = PHASES) -> None:
        self.states = {p: PhaseState(p) for p in phases}

    def set(
        self, phase: str, status: PhaseStatus, message: str = "", fraction: float | None = None
    ) -> bool:
        if phase not in self.states:
            return False
        ps = self.states[phase]
        ps.status, ps.message, ps.fraction = status, message, fraction
        return True

    def render(self) -> Any:
        from rich.table import Table
        from rich.text import Text

        t = Table(show_header=False, show_edge=False, box=None, padding=(0, 1, 0, 0))
        t.add_column(width=2, justify="center")
        t.add_column(width=15)
        t.add_column()
        for name, ps in self.states.items():
            label = Text(_LABELS.get(name, name), style="bold" if ps.status == "running" else "dim")
            body = ps.message
            if ps.fraction is not None:
                body = f"{ps.message}  {bar(ps.fraction)} {int(ps.fraction * 100):3d}%"
            t.add_row(_glyph_text(ps.status), label, Text(body))
        return t

    def plain_line(self, phase: str, status: PhaseStatus, message: str) -> str | None:
        from worldkernels.ui.theme import GLYPHS

        if status == "pending":
            return None
        g = GLYPHS.get(status)
        if g is None:
            return None
        return f"{g} {_LABELS.get(phase, phase):<15s} {message}"


def _gradient_bar_column(width: int = 28) -> Any:
    from rich.progress import ProgressColumn
    from rich.text import Text

    from worldkernels.ui.theme import ramp

    class _GradientBar(ProgressColumn):
        def render(self, task: Any) -> Any:
            total = task.total or 0
            frac = (task.completed / total) if total else 0.0
            filled = int(frac * width)
            stops = ramp()
            txt = Text()
            for i in range(width):
                if i < filled:
                    txt.append("━", style=_ramp_at(stops, i / max(1, width - 1)))
                else:
                    txt.append("╌", style="bright_black")
            return txt

    return _GradientBar()


class StepProgress:
    r"""Iteration progress for bench/run: gradient bar + live stat column (mode-accented)."""

    def __init__(self, total: int, label: str, mode: OutputMode) -> None:
        self.total = total
        self.label = label
        self.mode = mode
        self._progress: Any = None
        self._task: Any = None

    def __enter__(self) -> "StepProgress":
        if self.mode == OutputMode.AUTO:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            from worldkernels.ui.console import console
            from worldkernels.ui.theme import accent

            acc = accent()
            self._progress = Progress(
                SpinnerColumn(style=acc),
                TextColumn("{task.description}", style=acc),
                _gradient_bar_column(),
                TextColumn("{task.completed}/{task.total}", style="dim"),
                TextColumn("{task.fields[stat]}", style="dim"),
                TimeElapsedColumn(),
                console=console(),
                transient=True,
            )
            self._progress.__enter__()
            self._task = self._progress.add_task(self.label, total=self.total, stat="")
        return self

    def advance(self, stat: str = "") -> None:
        if self._progress is not None:
            self._progress.update(self._task, advance=1, stat=stat)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc, tb)
            self._progress = None


class _Working:
    def __init__(self, label: str, verbs: Sequence[str], counter: Callable[[], str] | None) -> None:
        self.label = label
        self.verbs = list(verbs) or [label]
        self.counter = counter
        self._t0 = time.monotonic()

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from rich.text import Text

        from worldkernels.ui.theme import accent

        el = time.monotonic() - self._t0
        frame = _SPIN[int(el * 10) % len(_SPIN)]
        verb = self.verbs[int(el / 2) % len(self.verbs)]
        t = Text()
        t.append(f"{frame} ", style=accent())
        t.append(f"{verb}… ", style=f"{accent()} bold")
        t.append(f"· {self.counter() if self.counter else f'{el:4.1f}s'}", style="dim")
        yield t


@contextmanager
def working(
    label: str,
    verbs: Sequence[str] | None = None,
    counter: Callable[[], str] | None = None,
) -> Iterator[Any]:
    r"""Animated working status: spinning glyph + rotating verb + live counter (TTY only)."""
    from worldkernels.ui.verbosity import resolve_output_mode

    if resolve_output_mode() != OutputMode.AUTO:
        yield None
        return

    from rich.live import Live

    from worldkernels.ui.console import console

    w = _Working(label, verbs or _VERBS, counter)
    live = Live(w, console=console(), refresh_per_second=12, transient=True)
    live.__enter__()
    try:
        yield w
    finally:
        live.__exit__(None, None, None)
