r"""Reusable terminal renderables: rules, tables, labeled fields, status lines, banner."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from worldkernels.ui.console import console, is_quiet_env


def rule(title: str = "", style: str = "rule") -> None:
    from rich.rule import Rule
    from rich.text import Text

    from worldkernels.ui.theme import style as theme_style

    con = console()
    line_style = theme_style(style)
    if title:
        con.print(
            Rule(
                Text(title, style=theme_style("heading")),
                characters="─",
                style=line_style,
                align="left",
            )
        )
    else:
        con.print(Rule(characters="─", style=line_style))


def table(*columns: str, title: str | None = None) -> Any:
    from rich.table import Table
    from rich.text import Text

    from worldkernels.ui.theme import style as theme_style

    head = theme_style("heading")
    t = Table(
        title=Text(title, style=head) if title else None,
        box=None,
        show_header=bool(columns),
        header_style=head,
        title_justify="left",
        pad_edge=False,
        padding=(0, 3, 0, 0),
    )
    for col in columns:
        t.add_column(col, overflow="fold")
    return t


def print_table(t: Any) -> None:
    console().print(t)


def field(
    label: str,
    value: Any,
    label_style: str | None = None,
    value_style: str | None = None,
) -> None:
    r"""Render a markup-safe ``label: value`` line with a colored label."""
    from rich.text import Text

    from worldkernels.ui.theme import style as theme_style

    t = Text()
    t.append(f"{label}: ", style=label_style or theme_style("key"))
    t.append(str(value), style=value_style)
    console().print(t)


def ok(msg: str) -> None:
    _glyph_line("done", msg)


def err(msg: str) -> None:
    _glyph_line("failed", msg)


def info(msg: str) -> None:
    _glyph_line("skipped", msg)


def line(text: str) -> None:
    console().print(text)


def _glyph_line(kind: str, msg: str) -> None:
    from rich.text import Text

    from worldkernels.ui.theme import GLYPHS, style as theme_style

    t = Text()
    t.append(f"{GLYPHS[kind]} ", style=theme_style(kind))
    t.append(msg)
    console().print(t)


@contextmanager
def status(msg: str) -> Iterator[None]:
    from worldkernels.ui.theme import style as theme_style

    con = console()
    if con.is_terminal and not is_quiet_env():
        with con.status(msg, spinner="dots", spinner_style=theme_style("accent")):
            yield
    else:
        yield


def metric_spark(label: str, values: Iterable[float], width: int = 40) -> None:
    r"""Print ``label: <ramp-tinted sparkline>`` for a series of values."""
    from rich.text import Text

    from worldkernels.ui.progress import sparkline
    from worldkernels.ui.theme import style as theme_style

    vals = list(values)
    if not vals:
        return
    t = Text()
    t.append(f"{label}: ", style=theme_style("key"))
    t.append_text(sparkline(vals, width))
    console().print(t)


def kv_rows(t: Any, pairs: Iterable[tuple[str, Any]], value_style: str | None = None) -> None:
    from rich.text import Text

    from worldkernels.ui.theme import style as theme_style

    key_style = theme_style("key")
    for k, v in pairs:
        t.add_row(Text(str(k), style=key_style), Text(str(v), style=value_style or ""))


_BANNER_ROWS = ("╦ ╦╦╔═", "║║║╠╩╗", "╚╩╝╩ ╩")
_BANNER_HI = "#ECFCCB bold"


def _banner_text(ver: str, sweep: int | None) -> Any:
    from rich.text import Text

    from worldkernels.ui.theme import BANNER_RAMP

    logo = Text()
    for r, (row, base) in enumerate(zip(_BANNER_ROWS, BANNER_RAMP)):
        for c, ch in enumerate(row):
            logo.append(ch, style=_BANNER_HI if sweep == c else base)
        if r == 1:
            logo.append("   worldkernels", style="bold")
            logo.append(f"  v{ver}" if ver else "", style="dim")
        elif r == 2:
            logo.append("   GPU-first world model simulation engine", style="dim")
        logo.append("\n")
    return logo


def banner() -> None:
    con = console()
    if not con.is_terminal:
        return
    import time

    from rich.live import Live

    try:
        from worldkernels import __version__ as ver
    except Exception:
        ver = ""

    with Live(_banner_text(ver, None), console=con, refresh_per_second=30) as live:
        for s in range(len(_BANNER_ROWS[0]) + 2):
            live.update(_banner_text(ver, s if s < len(_BANNER_ROWS[0]) else None))
            time.sleep(0.045)
        live.update(_banner_text(ver, None))
