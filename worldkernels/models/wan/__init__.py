r"""Wan-family video-diffusion pipelines (diffusers-backed)."""

from __future__ import annotations

import importlib as _importlib

__all__ = ["WanI2VPipeline", "WanLatent"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WanI2VPipeline": ("worldkernels.models.wan.pipeline_wan_i2v", "WanI2VPipeline"),
    "WanLatent": ("worldkernels.models.wan.pipeline_wan_i2v", "WanLatent"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
