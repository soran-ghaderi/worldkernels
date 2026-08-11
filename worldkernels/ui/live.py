r"""rich.Live driver over a renderable-provider callable, with mode degradation."""

from __future__ import annotations

from typing import Any, Callable

from worldkernels.ui.verbosity import OutputMode


class LiveDisplay:
    r"""Drive in-place updates from a ``renderable()`` provider.

    In AUTO (tty) mode it wraps ``rich.Live``; in PLAIN/JSON/QUIET it stays inert and
    the caller emits its own line/json/nothing. Single live renderer, no duplication.
    """

    def __init__(
        self,
        renderable: Callable[[], Any],
        mode: OutputMode,
        refresh_per_second: int = 10,
        transient: bool = False,
    ) -> None:
        self._renderable = renderable
        self._mode = mode
        self._refresh = refresh_per_second
        self._transient = transient
        self._live: Any = None

    @property
    def active(self) -> bool:
        return self._live is not None

    def __enter__(self) -> "LiveDisplay":
        if self._mode == OutputMode.AUTO:
            try:
                from rich.live import Live

                from worldkernels.ui.console import console

                self._live = Live(
                    self._renderable(),
                    console=console(),
                    refresh_per_second=self._refresh,
                    transient=self._transient,
                )
                self._live.__enter__()
            except ImportError:
                self._live = None
        return self

    def update(self) -> None:
        if self._live is not None:
            self._live.update(self._renderable())

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._live is not None:
            self._live.update(self._renderable(), refresh=True)
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
