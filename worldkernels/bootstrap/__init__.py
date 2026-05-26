r"""Lazy model bootstrap: resolve ref, provision deps, fetch weights."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "prepare",
    "PreparedModel",
    "ProgressController",
    "BootstrapError",
    "FetchDisabledError",
    "AuthRequiredError",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "prepare": ("worldkernels.bootstrap.pipeline", "prepare"),
    "PreparedModel": ("worldkernels.bootstrap.pipeline", "PreparedModel"),
    "ProgressController": ("worldkernels.bootstrap.progress", "ProgressController"),
    "BootstrapError": ("worldkernels.bootstrap.errors", "BootstrapError"),
    "FetchDisabledError": ("worldkernels.bootstrap.errors", "FetchDisabledError"),
    "AuthRequiredError": ("worldkernels.bootstrap.errors", "AuthRequiredError"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
