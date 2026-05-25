r"""Pipeline backed by NVIDIA's cosmos_predict2 package.

Wraps cosmos_predict2 for inference: dep-stub injection, package location, and
a pipeline that drives model loading, denoising, and VAE decode. Lives under
``worldkernels.models`` as one model family alongside ``wan``.
"""

import importlib as _importlib

__all__ = [
    "CosmosPredict2Latent",
    "CosmosPredict2Pipeline",
    "ensure_cosmos_predict2",
]

_PIPELINE = "worldkernels.models.cosmos_predict2.pipeline"
_DEPS = "worldkernels.models.cosmos_predict2.deps"

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CosmosPredict2Latent": (_PIPELINE, "CosmosPredict2Latent"),
    "CosmosPredict2Pipeline": (_PIPELINE, "CosmosPredict2Pipeline"),
    "ensure_cosmos_predict2": (_DEPS, "ensure_cosmos_predict2"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
