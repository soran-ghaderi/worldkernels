r"""logging-to-handler pipeline gated to CLI verbosity."""

from __future__ import annotations

import logging

from worldkernels.ui.verbosity import Verbosity

_LEVELS = {
    Verbosity.QUIET: logging.ERROR,
    Verbosity.NORMAL: logging.WARNING,
    Verbosity.VERBOSE: logging.INFO,
    Verbosity.DEBUG: logging.DEBUG,
}

_PLAIN_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_PLAIN_DATEFMT = "%m-%d %H:%M:%S"


def configure_logging(verbosity: Verbosity) -> None:
    r"""Install a single handler on the ``worldkernels`` logger at the mapped level.

    TTY sessions get a RichHandler; plain/json/quiet output modes get a
    compact line format (``INFO 07-12 10:00:00 [serve.py:42] msg``).
    """
    from worldkernels.ui.verbosity import OutputMode, resolve_output_mode

    level = _LEVELS[verbosity]
    root = logging.getLogger("worldkernels")
    root.handlers.clear()
    handler: logging.Handler
    if resolve_output_mode() == OutputMode.AUTO:
        from rich.logging import RichHandler

        from worldkernels.ui.console import console

        handler = RichHandler(
            console=console(),
            show_path=False,
            rich_tracebacks=True,
            markup=False,
            show_time=verbosity >= Verbosity.DEBUG,
        )
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_PLAIN_DATEFMT))
    handler.setLevel(level)
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
