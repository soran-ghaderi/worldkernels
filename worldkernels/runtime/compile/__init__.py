r"""Compilation backends: regional torch.compile and CUDA graph capture."""

from __future__ import annotations

import importlib as _importlib

__all__ = ["regionally_compile", "CUDAGraphRunner"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "regionally_compile": ("worldkernels.runtime.compile.torch_compile", "regionally_compile"),
    "CUDAGraphRunner": ("worldkernels.runtime.compile.cuda_graph", "CUDAGraphRunner"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
