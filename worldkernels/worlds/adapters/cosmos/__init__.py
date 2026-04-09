r"""Cosmos-Predict2.5 video-to-world adapter."""

import importlib as _importlib

__all__ = ["CosmosPredict2World", "CosmosLatent"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CosmosPredict2World": ("worldkernels.worlds.adapters.cosmos.adapter", "CosmosPredict2World"),
    "CosmosLatent": ("worldkernels.worlds.adapters._cosmos_predict2", "CosmosLatent"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
