r"""Ulysses sequence parallelism.

Ulysses shards the sequence dimension across ranks for the linear layers and,
inside attention, transposes that sharding onto the head dimension with an
all-to-all so each rank computes full-sequence attention for a slice of the
heads. Two all-to-alls bracket the attention call: sequence-sharded ->
head-sharded before, head-sharded -> sequence-sharded after.
"""

from __future__ import annotations

import torch

from worldkernels.distributed.parallel_state import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
)

__all__ = ["ulysses_all_to_all"]


def ulysses_all_to_all(
    x: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    r"""All-to-all redistribution across the sequence-parallel group.

    Splits ``x`` into equal chunks along ``scatter_dim``, exchanges one chunk
    with every rank, and concatenates the received chunks along ``gather_dim``.
    Identity when sequence-parallel world size is 1.

    Args:
        x: Tensor sharded along ``scatter_dim``.
        scatter_dim: Dimension to split and send.
        gather_dim: Dimension to concatenate received chunks on.
    """
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return x
    import torch.distributed as dist

    if x.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"scatter dim {scatter_dim} (size {x.shape[scatter_dim]}) not divisible "
            f"by sequence-parallel world size {world_size}"
        )
    send = [c.contiguous() for c in x.chunk(world_size, dim=scatter_dim)]
    recv = [torch.empty_like(send[0]) for _ in range(world_size)]
    dist.all_to_all(recv, send, group=get_sequence_parallel_group())
    return torch.cat(recv, dim=gather_dim)
