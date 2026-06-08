r"""logging-to-RichHandler pipeline gated to CLI verbosity."""

from __future__ import annotations

import logging

from worldkernels.ui.verbosity import Verbosity

_LEVELS = {
    Verbosity.QUIET: logging.ERROR,
    Verbosity.NORMAL: logging.WARNING,
    Verbosity.VERBOSE: logging.INFO,
    Verbosity.DEBUG: logging.DEBUG,
}


def configure_logging(verbosity: Verbosity) -> None:
    r"""Install a single RichHandler on the ``worldkernels`` logger at the mapped level."""
    from rich.logging import RichHandler

    from worldkernels.ui.console import console

    level = _LEVELS[verbosity]
    root = logging.getLogger("worldkernels")
    root.handlers.clear()
    handler = RichHandler(
        console=console(),
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        show_time=verbosity >= Verbosity.DEBUG,
    )
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
