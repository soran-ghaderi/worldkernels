r"""Signature palette: lime brand, per-mode accents, glyphs, and gradient ramps.

Content text stays the terminal's default fg (legible on light and dark); the accent
only decorates (glyphs, rules, bars, spinners, headings). The active mode picks the
accent — set once per command via `set_mode`. Truecolor hex; rich downsamples for
256/16-color and strips under NO_COLOR.
"""

from __future__ import annotations

from typing import Final

MODES: Final[dict[str, str]] = {
    "serve": "#84CC16",
    "generate": "#A855F7",
    "measure": "#F59E0B",
    "fetch": "#14B8A6",
    "read": "#94A3B8",
    "danger": "#F43F5E",
}

MODE_RAMPS: Final[dict[str, tuple[str, str, str]]] = {
    "serve": ("#A3E635", "#22C55E", "#14B8A6"),
    "generate": ("#C084FC", "#A855F7", "#7C3AED"),
    "measure": ("#FCD34D", "#F59E0B", "#D97706"),
    "fetch": ("#5EEAD4", "#14B8A6", "#0D9488"),
    "read": ("#CBD5E1", "#94A3B8", "#64748B"),
    "danger": ("#FB7185", "#F43F5E", "#BE123C"),
}

_SEMANTIC: Final[dict[str, str]] = {
    "success": "#16A34A",
    "error": "#E5484D",
    "warn": "#F59E0B",
    "info": "dim",
    "muted": "dim",
    "metric": "",
}

GLYPHS: Final[dict[str, str]] = {
    "done": "✓",
    "failed": "✗",
    "running": "▸",
    "pending": "·",
    "skipped": "·",
}

STATUS_STYLE: Final[dict[str, str]] = {
    "done": "success",
    "failed": "error",
    "running": "accent",
    "pending": "muted",
    "skipped": "muted",
}

BANNER_RAMP: Final[tuple[str, ...]] = ("#A3E635 bold", "#84CC16 bold", "#14B8A6 bold")

_DEFAULT_MODE: Final[str] = "serve"
_MODE: str = _DEFAULT_MODE


def set_mode(mode: str) -> None:
    global _MODE
    _MODE = mode if mode in MODES else _DEFAULT_MODE


def get_mode() -> str:
    return _MODE


def accent() -> str:
    r"""Active mode's accent hex."""
    return MODES[_MODE]


def ramp() -> tuple[str, str, str]:
    r"""Active mode's 3-stop gradient."""
    return MODE_RAMPS[_MODE]


def style(name: str) -> str:
    r"""Resolve a semantic/status/accent name to a rich style; pass raw styles through.

    `accent`/`heading`/`key`/`rule` follow the active mode; semantic names are fixed.
    """
    if name in ("accent", "heading"):
        return f"{accent()} bold"
    if name in ("key", "rule"):
        return accent()
    if name in _SEMANTIC:
        return _SEMANTIC[name]
    if name in STATUS_STYLE:
        return style(STATUS_STYLE[name])
    return name
