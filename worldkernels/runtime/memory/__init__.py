r"""Memory subsystem: block allocation, KV paging, buffer pools, offloading.

``TrajectoryCache`` is torch-free and eagerly exported; the tensor-backed
managers are lazily imported.
"""

from __future__ import annotations

import importlib as _importlib

from worldkernels.runtime.memory.trajectory_cache import PrefixMatch, TrajectoryCache

__all__ = [
    "BlockManager",
    "LatentPool",
    "Offloader",
    "MemoryTier",
    "KVCacheManager",
    "TrajectoryCache",
    "PrefixMatch",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BlockManager": ("worldkernels.runtime.memory.block_manager", "BlockManager"),
    "LatentPool": ("worldkernels.runtime.memory.latent_pool", "LatentPool"),
    "Offloader": ("worldkernels.runtime.memory.offload", "Offloader"),
    "MemoryTier": ("worldkernels.runtime.memory.offload", "MemoryTier"),
    "KVCacheManager": ("worldkernels.runtime.memory.kv_cache", "KVCacheManager"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
