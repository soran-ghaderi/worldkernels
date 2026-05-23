r"""Distributed-execution primitives: parallel state, collectives, CFG/sequence parallelism.

Parallel-state queries are torch-free and eagerly exported; the collective and
parallel-attention helpers import torch and are resolved lazily.
"""

from __future__ import annotations

import importlib as _importlib

from worldkernels.distributed.parallel_state import (
    destroy_distributed,
    get_cfg_parallel_group,
    get_cfg_parallel_rank,
    get_cfg_parallel_world_size,
    get_global_rank,
    get_parallel_config,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    get_world_size,
    init_distributed,
    is_initialized,
)

__all__ = [
    "init_distributed",
    "destroy_distributed",
    "is_initialized",
    "get_parallel_config",
    "get_global_rank",
    "get_world_size",
    "get_tensor_parallel_rank",
    "get_tensor_parallel_world_size",
    "get_tensor_parallel_group",
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sequence_parallel_group",
    "get_cfg_parallel_rank",
    "get_cfg_parallel_world_size",
    "get_cfg_parallel_group",
    "tensor_parallel_all_reduce",
    "tensor_parallel_all_gather",
    "CFGParallelMixin",
    "ulysses_all_to_all",
    "ring_rotate",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "tensor_parallel_all_reduce": (
        "worldkernels.distributed.communication",
        "tensor_parallel_all_reduce",
    ),
    "tensor_parallel_all_gather": (
        "worldkernels.distributed.communication",
        "tensor_parallel_all_gather",
    ),
    "CFGParallelMixin": ("worldkernels.distributed.cfg_parallel", "CFGParallelMixin"),
    "ulysses_all_to_all": (
        "worldkernels.distributed.sequence_parallel",
        "ulysses_all_to_all",
    ),
    "ring_rotate": ("worldkernels.distributed.sequence_parallel", "ring_rotate"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
