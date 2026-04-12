r"""Shared internals for the cosmos_predict2 model family (cosmos, dreamdojo)."""

import importlib as _importlib

__all__ = [
    "CosmosBaseWorld",
    "CosmosLatent",
    "_LATENT_CH",
    "_SPATIAL_FACTOR",
    "download_dreamdojo_checkpoint",
    "download_hf_file",
    "ensure_cosmos_predict2",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CosmosBaseWorld": ("worldkernels.worlds.adapters._cosmos_predict2._base", "CosmosBaseWorld"),
    "CosmosLatent": ("worldkernels.worlds.adapters._cosmos_predict2._base", "CosmosLatent"),
    "_LATENT_CH": ("worldkernels.worlds.adapters._cosmos_predict2._base", "_LATENT_CH"),
    "_SPATIAL_FACTOR": ("worldkernels.worlds.adapters._cosmos_predict2._base", "_SPATIAL_FACTOR"),
    "download_dreamdojo_checkpoint": (
        "worldkernels.worlds.adapters._cosmos_predict2._base",
        "download_dreamdojo_checkpoint",
    ),
    "download_hf_file": ("worldkernels.worlds.adapters._cosmos_predict2._base", "download_hf_file"),
    "ensure_cosmos_predict2": (
        "worldkernels.worlds.adapters._cosmos_predict2._deps",
        "ensure_cosmos_predict2",
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
