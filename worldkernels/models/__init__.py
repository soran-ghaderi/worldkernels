r"""Native video-diffusion model families and their pipelines.

Mirrors vLLM-Omni's ``diffusion/models/`` layout: one subpackage per model
family (``wan/``, ...), a shared ``schedulers/`` package, and a ``registry``
mapping family names to pipeline classes. Adding a model is a new subpackage
plus one ``registry`` row.

The ``registry`` module is import-light (no torch); pipeline modules are not
and are resolved lazily.
"""

from __future__ import annotations

from worldkernels.models.registry import (
    get_pipeline_class,
    list_pipelines,
    register_pipeline,
)

__all__ = ["get_pipeline_class", "list_pipelines", "register_pipeline"]
