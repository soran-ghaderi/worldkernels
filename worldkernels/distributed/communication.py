r"""Collective communication wrappers.

Every collective is an identity no-op when its axis has world size 1, so
single-GPU code paths carry no distributed overhead and need no guards at the
call site.
"""

from __future__ import annotations

import torch

from worldkernels.distributed.parallel_state import (
    get_tensor_parallel_group,
    get_tensor_parallel_world_size,
)

__all__ = [
    "tensor_parallel_all_reduce",
    "tensor_parallel_all_gather",
]


def tensor_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    r"""Sum ``tensor`` across the tensor-parallel group, in place."""
    if get_tensor_parallel_world_size() == 1:
        return tensor
    import torch.distributed as dist

    dist.all_reduce(tensor, group=get_tensor_parallel_group())
    return tensor


def tensor_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Gather ``tensor`` across the tensor-parallel group, concatenating on ``dim``."""
    world_size = get_tensor_parallel_world_size()
    if world_size == 1:
        return tensor
    import torch.distributed as dist

    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor.contiguous(), group=get_tensor_parallel_group())
    return torch.cat(shards, dim=dim)
