r"""DreamDojo model family: native action-conditioned DiT, tokenizer, checkpoints."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "ActionVideoDiT",
    "NET_2B",
    "NET_14B",
    "VideoTokenizer",
    "download_dreamdojo_checkpoint",
    "LatentActionEncoder",
    "load_net",
    "remap_state_dict",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ActionVideoDiT": ("worldkernels.models.dreamdojo.net", "ActionVideoDiT"),
    "NET_2B": ("worldkernels.models.dreamdojo.net", "NET_2B"),
    "NET_14B": ("worldkernels.models.dreamdojo.net", "NET_14B"),
    "VideoTokenizer": ("worldkernels.models.dreamdojo.tokenizer", "VideoTokenizer"),
    "download_dreamdojo_checkpoint": (
        "worldkernels.models.dreamdojo.checkpoint",
        "download_dreamdojo_checkpoint",
    ),
    "LatentActionEncoder": ("worldkernels.models.dreamdojo.lam", "LatentActionEncoder"),
    "load_net": ("worldkernels.models.dreamdojo.checkpoint", "load_net"),
    "remap_state_dict": ("worldkernels.models.dreamdojo.checkpoint", "remap_state_dict"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
