r"""Attention backend implementations."""

from __future__ import annotations

import importlib as _importlib

__all__ = ["SDPABackend", "FlashAttentionBackend"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "SDPABackend": ("worldkernels.runtime.attention.backends.sdpa", "SDPABackend"),
    "FlashAttentionBackend": (
        "worldkernels.runtime.attention.backends.flash",
        "FlashAttentionBackend",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
