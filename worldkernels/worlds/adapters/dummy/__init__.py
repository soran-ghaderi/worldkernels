r"""Dummy world model adapter for development and testing."""

import importlib as _importlib

__all__ = ["DummyWorld"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DummyWorld": ("worldkernels.worlds.adapters.dummy.adapter", "DummyWorld"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
