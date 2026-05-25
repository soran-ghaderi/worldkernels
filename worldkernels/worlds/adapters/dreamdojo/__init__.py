r"""DreamDojo action-conditioned world model adapter."""

import importlib as _importlib

__all__ = ["DreamDojoWorld"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DreamDojoWorld": ("worldkernels.worlds.adapters.dreamdojo.adapter", "DreamDojoWorld"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
