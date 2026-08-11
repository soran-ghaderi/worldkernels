r"""Schedulers for video-diffusion pipelines."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "FlowUniPCMultistepScheduler",
    "TRIGFLOW_TIMES_4STEP",
    "trigflow_time",
    "trigflow_scaling",
    "trigflow_step",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlowUniPCMultistepScheduler": (
        "worldkernels.models.schedulers.flow_unipc",
        "FlowUniPCMultistepScheduler",
    ),
    "TRIGFLOW_TIMES_4STEP": ("worldkernels.models.schedulers.trigflow", "TRIGFLOW_TIMES_4STEP"),
    "trigflow_time": ("worldkernels.models.schedulers.trigflow", "trigflow_time"),
    "trigflow_scaling": ("worldkernels.models.schedulers.trigflow", "trigflow_scaling"),
    "trigflow_step": ("worldkernels.models.schedulers.trigflow", "trigflow_step"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
