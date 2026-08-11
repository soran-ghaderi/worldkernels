r"""Process-global CLI verbosity and output mode."""

from __future__ import annotations

import enum
import os


class Verbosity(enum.IntEnum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class OutputMode(enum.Enum):
    AUTO = "auto"
    PLAIN = "plain"
    JSON = "json"
    QUIET = "quiet"


_VERBOSITY: Verbosity = Verbosity.NORMAL
_OUTPUT: OutputMode = OutputMode.AUTO


def get_verbosity() -> Verbosity:
    return _VERBOSITY


def set_verbosity(v: Verbosity) -> None:
    global _VERBOSITY
    _VERBOSITY = v
    if v == Verbosity.QUIET:
        os.environ["WORLDKERNELS_QUIET"] = "1"


def get_output_mode() -> OutputMode:
    return _OUTPUT


def set_output_mode(m: OutputMode) -> None:
    global _OUTPUT
    _OUTPUT = m


def resolve_output_mode() -> OutputMode:
    r"""Collapse AUTO to QUIET/PLAIN/AUTO using env, verbosity, and tty state."""
    from worldkernels.ui.console import console, is_quiet_env

    if is_quiet_env() or _VERBOSITY == Verbosity.QUIET:
        return OutputMode.QUIET
    if _OUTPUT != OutputMode.AUTO:
        return _OUTPUT
    return OutputMode.AUTO if console().is_terminal else OutputMode.PLAIN
