r"""Process-level active RuntimeConfig.

Bake-time gates (torch_compile, quantization, cuda_graphs at warmup) read the
active config here instead of threading it through every world/pipeline
signature. The engine sets it at construction. Step-time, per-session toggles
ride `ForwardContext` instead (set per request).

One engine per process is the common case; a second engine overwrites the
active config (documented limitation — per-session overrides cover the
multi-config-in-one-process case at step time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.config.runtime import RuntimeConfig

_ACTIVE: "RuntimeConfig | None" = None


def set_active_config(config: "RuntimeConfig") -> None:
    global _ACTIVE
    _ACTIVE = config


def get_active_config() -> "RuntimeConfig":
    global _ACTIVE
    if _ACTIVE is None:
        from worldkernels.config.runtime import RuntimeConfig

        _ACTIVE = RuntimeConfig()
    return _ACTIVE


def clear_active_config() -> None:
    global _ACTIVE
    _ACTIVE = None
