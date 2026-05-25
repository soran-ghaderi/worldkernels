r"""DreamDojo model family: checkpoint download and DCP-to-pt conversion."""

from __future__ import annotations

import importlib as _importlib

__all__ = ["download_dreamdojo_checkpoint"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "download_dreamdojo_checkpoint": (
        "worldkernels.models.dreamdojo.checkpoint",
        "download_dreamdojo_checkpoint",
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
