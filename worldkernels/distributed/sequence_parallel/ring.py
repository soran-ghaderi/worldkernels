r"""Ring sequence parallelism.

Ring attention keeps the sequence sharded across ranks and circulates each
rank's key/value block around a ring, accumulating attention with an online
softmax so no rank ever materializes the full sequence. `ring_rotate()` is
the communication primitive — one hop around the ring; the online-softmax
accumulation loop lives in the attention backend that calls it.
"""

from __future__ import annotations

import torch

from worldkernels.distributed.parallel_state import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)

__all__ = ["ring_rotate"]


def ring_rotate(tensor: torch.Tensor) -> torch.Tensor:
    r"""Send ``tensor`` to the next rank and return the tensor from the previous one.

    One hop of the ring: rank \(r\) sends to \(r{+}1\) and receives from
    \(r{-}1\) (modulo the ring size). Identity when sequence-parallel world
    size is 1.
    """
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor
    import torch.distributed as dist

    group = get_sequence_parallel_group()
    rank = get_sequence_parallel_rank()
    ranks = dist.get_process_group_ranks(group)
    send_to = ranks[(rank + 1) % world_size]
    recv_from = ranks[(rank - 1) % world_size]

    recv = torch.empty_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor.contiguous(), send_to, group),
        dist.P2POp(dist.irecv, recv, recv_from, group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return recv
