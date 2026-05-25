r"""Registry of video-diffusion pipeline classes, keyed by model family.

Adding a model family: drop a ``pipeline_*.py`` under ``models/<family>/`` that
defines the pipeline class, then add one row to ``_PIPELINES``. Resolution is
lazy, so importing this registry never imports torch or diffusers.
"""

from __future__ import annotations

import importlib

__all__ = ["get_pipeline_class", "list_pipelines", "register_pipeline"]

_PIPELINES: dict[str, tuple[str, str]] = {
    "wan_i2v": ("worldkernels.models.wan.pipeline_wan_i2v", "WanI2VPipeline"),
    "cosmos_predict2": ("worldkernels.models.cosmos_predict2.pipeline", "CosmosPredict2Pipeline"),
}


def register_pipeline(name: str, module: str, class_name: str) -> None:
    r"""Register a pipeline family ``name`` resolving to ``module:class_name``."""
    _PIPELINES[name] = (module, class_name)


def list_pipelines() -> list[str]:
    return sorted(_PIPELINES)


def get_pipeline_class(name: str) -> type:
    r"""Resolve a pipeline family name to its class, importing the module lazily."""
    if name not in _PIPELINES:
        available = ", ".join(sorted(_PIPELINES)) or "(none)"
        raise KeyError(f"Unknown pipeline {name!r}. Available: {available}")
    module_path, class_name = _PIPELINES[name]
    return getattr(importlib.import_module(module_path), class_name)
