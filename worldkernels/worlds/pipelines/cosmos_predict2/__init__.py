r"""Pipeline backed by NVIDIA's cosmos_predict2 package.

Wraps cosmos_predict2 for inference: dep-stub injection, package location, and
a pipeline that drives model loading, denoising, and VAE decode. Composed by
the cosmos and dreamdojo adapters until Phase 5 lands a native pipeline at
``worldkernels.worlds.pipelines.video_diffusion``.
"""

import importlib as _importlib

__all__ = [
    "CosmosPredict2Latent",
    "CosmosPredict2Pipeline",
    "ensure_cosmos_predict2",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CosmosPredict2Latent": ("worldkernels.worlds.pipelines.cosmos_predict2.pipeline", "CosmosPredict2Latent"),
    "CosmosPredict2Pipeline": (
        "worldkernels.worlds.pipelines.cosmos_predict2.pipeline",
        "CosmosPredict2Pipeline",
    ),
    "ensure_cosmos_predict2": ("worldkernels.worlds.pipelines.cosmos_predict2.deps", "ensure_cosmos_predict2"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
