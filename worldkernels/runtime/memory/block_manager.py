r"""Fixed-size block (slab) allocator for latent and KV storage.

Pre-allocates one contiguous slab on the device and hands out block indices,
so steady-state operation never calls ``cudaMalloc`` / ``cudaFree``. Blocks are
reference-counted, which makes copy-on-write sharing (session branching,
trajectory-prefix reuse) a refcount bump rather than a copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["BlockManager"]


class BlockManager:
    r"""Slab allocator handing out fixed-shape blocks of a pre-allocated tensor.

    Args:
        block_shape: Shape of one block (the per-block trailing dims).
        num_blocks: Number of blocks in the slab.
        device: Device the slab lives on.
        dtype: Slab dtype.
    """

    def __init__(
        self,
        *,
        block_shape: "Sequence[int]",
        num_blocks: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.block_shape = tuple(block_shape)
        self.num_blocks = num_blocks
        self.device = device
        self.dtype = dtype
        self._slab = torch.empty((num_blocks, *self.block_shape), device=device, dtype=dtype)
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._refcount: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self, n_blocks: int = 1) -> list[int]:
        r"""Reserve ``n_blocks`` blocks, returning their ids (refcount 1)."""
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks > len(self._free):
            raise MemoryError(
                f"block pool exhausted: requested {n_blocks}, {len(self._free)} free "
                f"of {self.num_blocks}"
            )
        ids = [self._free.pop() for _ in range(n_blocks)]
        for block_id in ids:
            self._refcount[block_id] = 1
        return ids

    def block(self, block_id: int) -> torch.Tensor:
        r"""Return the tensor view of block ``block_id``."""
        return self._slab[block_id]

    def share(self, block_ids: "Sequence[int]") -> list[int]:
        r"""Copy-on-write share: bump refcounts and return the same ids."""
        for block_id in block_ids:
            self._refcount[block_id] += 1
        return list(block_ids)

    def free(self, block_ids: "Sequence[int]") -> None:
        r"""Release a reference to each block; return it to the free list at zero."""
        for block_id in block_ids:
            count = self._refcount.get(block_id, 0)
            if count <= 0:
                continue
            if count == 1:
                del self._refcount[block_id]
                self._free.append(block_id)
            else:
                self._refcount[block_id] = count - 1

    def refcount(self, block_id: int) -> int:
        return self._refcount.get(block_id, 0)

    def reset(self) -> None:
        r"""Release every block."""
        self._free = list(reversed(range(self.num_blocks)))
        self._refcount.clear()
