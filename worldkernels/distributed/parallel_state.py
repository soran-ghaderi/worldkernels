r"""Distributed parallel-state registry.

Holds each rank's position within every parallelism axis (tensor, sequence,
CFG). The world is the product mesh

\[ W = \text{DP} \cdot \text{PP} \cdot \text{SP} \cdot \text{CFG} \cdot \text{TP} \]

with TP innermost (adjacent global ranks share a TP group). Single-rank
execution is the all-identity special case: every getter returns rank 0 /
world size 1 / group ``None``, and no ``torch.distributed`` call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from worldkernels.config.parallel_config import ParallelConfig

if TYPE_CHECKING:
    import torch.distributed as dist  # noqa: F401

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
]


@dataclass
class _GroupInfo:
    r"""This rank's coordinates within one parallelism axis."""

    rank: int = 0
    world_size: int = 1
    ranks: list[int] = field(default_factory=lambda: [0])
    group: Any = None


@dataclass
class _ParallelState:
    config: ParallelConfig = field(default_factory=ParallelConfig)
    global_rank: int = 0
    world_size: int = 1
    tensor: _GroupInfo = field(default_factory=_GroupInfo)
    sequence: _GroupInfo = field(default_factory=_GroupInfo)
    cfg: _GroupInfo = field(default_factory=_GroupInfo)
    initialized: bool = False


_STATE = _ParallelState()


def init_distributed(
    config: ParallelConfig,
    *,
    backend: str = "nccl",
) -> None:
    r"""Initialize process groups for the parallel ``config``.

    Single-rank configs initialize identity groups without touching
    ``torch.distributed``. Multi-rank configs initialize the process group
    (rank and world size read from the ``RANK`` / ``WORLD_SIZE`` environment)
    and build the per-axis sub-groups.
    """
    _STATE.config = config
    _STATE.world_size = config.world_size

    if config.world_size == 1:
        _STATE.global_rank = 0
        _STATE.tensor = _GroupInfo()
        _STATE.sequence = _GroupInfo()
        _STATE.cfg = _GroupInfo()
        _STATE.initialized = True
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    global_rank = int(os.environ.get("RANK", dist.get_rank()))
    if dist.get_world_size() != config.world_size:
        raise ValueError(
            f"launched world size {dist.get_world_size()} != config world size {config.world_size}"
        )
    _STATE.global_rank = global_rank

    tp = config.tensor_parallel_size
    sp = config.sequence_parallel_size
    cfg = config.cfg_parallel_size

    _STATE.tensor = _build_axis(global_rank, inner=1, axis=tp, label="tp")
    _STATE.cfg = _build_axis(global_rank, inner=tp, axis=cfg, label="cfg")
    _STATE.sequence = _build_axis(global_rank, inner=tp * cfg, axis=sp, label="sp")
    _STATE.initialized = True


def _build_axis(global_rank: int, *, inner: int, axis: int, label: str) -> _GroupInfo:
    import torch.distributed as dist

    outer = _STATE.world_size // (inner * axis)
    info = _GroupInfo(world_size=axis)
    for o in range(outer):
        for i in range(inner):
            seed = o * inner * axis + i
            ranks = [seed + a * inner for a in range(axis)]
            group = dist.new_group(ranks) if axis > 1 else None
            if global_rank in ranks:
                info.ranks = ranks
                info.rank = ranks.index(global_rank)
                info.group = group
    return info


def destroy_distributed() -> None:
    r"""Tear down process groups and reset to single-rank identity."""
    if _STATE.world_size > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    _STATE.__dict__.update(_ParallelState().__dict__)


def is_initialized() -> bool:
    return _STATE.initialized


def get_parallel_config() -> ParallelConfig:
    return _STATE.config


def get_global_rank() -> int:
    return _STATE.global_rank


def get_world_size() -> int:
    return _STATE.world_size


def get_tensor_parallel_rank() -> int:
    return _STATE.tensor.rank


def get_tensor_parallel_world_size() -> int:
    return _STATE.tensor.world_size


def get_tensor_parallel_group() -> Any:
    return _STATE.tensor.group


def get_sequence_parallel_rank() -> int:
    return _STATE.sequence.rank


def get_sequence_parallel_world_size() -> int:
    return _STATE.sequence.world_size


def get_sequence_parallel_group() -> Any:
    return _STATE.sequence.group


def get_cfg_parallel_rank() -> int:
    return _STATE.cfg.rank


def get_cfg_parallel_world_size() -> int:
    return _STATE.cfg.world_size


def get_cfg_parallel_group() -> Any:
    return _STATE.cfg.group
