r"""Pluggable attention backends and selection."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "SDPABackend",
    "FlashAttentionBackend",
    "select_attention_backend",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AttentionBackend": ("worldkernels.runtime.attention.backend", "AttentionBackend"),
    "AttentionMetadata": ("worldkernels.runtime.attention.backend", "AttentionMetadata"),
    "SDPABackend": ("worldkernels.runtime.attention.backends.sdpa", "SDPABackend"),
    "FlashAttentionBackend": (
        "worldkernels.runtime.attention.backends.flash",
        "FlashAttentionBackend",
    ),
    "select_attention_backend": (
        "worldkernels.runtime.attention.selector",
        "select_attention_backend",
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
