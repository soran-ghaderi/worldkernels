r"""Paged KV cache for causal world models.

Each session's KV cache is a chain of fixed-size blocks drawn from a
`BlockManager`. Branching a
session forks the block table copy-on-write: blocks are shared (a refcount
bump) until one branch writes past the divergence point.

Only causal worlds (``transition_mode == CAUSAL``) hold a KV cache;
bidirectional worlds pay none of this cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.runtime.memory.block_manager import BlockManager

__all__ = ["KVCacheManager"]


class KVCacheManager:
    r"""Per-session paged KV-cache block tables over a shared block pool.

    Args:
        block_manager: The block pool KV blocks are drawn from.
    """

    def __init__(self, block_manager: "BlockManager") -> None:
        self.blocks = block_manager
        self._tables: dict[str, list[int]] = {}

    def allocate(self, session_id: str, num_blocks: int) -> list[int]:
        r"""Allocate ``num_blocks`` KV blocks for ``session_id``."""
        ids = self.blocks.allocate(num_blocks)
        self._tables[session_id] = list(ids)
        return ids

    def append(self, session_id: str, num_blocks: int = 1) -> list[int]:
        r"""Grow a session's KV table by ``num_blocks`` (one decode chunk)."""
        if session_id not in self._tables:
            raise KeyError(f"no KV table for session {session_id!r}")
        ids = self.blocks.allocate(num_blocks)
        self._tables[session_id].extend(ids)
        return ids

    def block_table(self, session_id: str) -> list[int]:
        r"""Return the ordered block ids backing a session's KV cache."""
        return list(self._tables.get(session_id, []))

    def fork(self, src_session_id: str, dst_session_id: str) -> list[int]:
        r"""Copy-on-write fork: ``dst`` shares ``src``'s blocks until it writes."""
        if src_session_id not in self._tables:
            raise KeyError(f"no KV table for session {src_session_id!r}")
        shared = self.blocks.share(self._tables[src_session_id])
        self._tables[dst_session_id] = shared
        return shared

    def free(self, session_id: str) -> None:
        r"""Release a session's KV blocks."""
        table = self._tables.pop(session_id, None)
        if table:
            self.blocks.free(table)
