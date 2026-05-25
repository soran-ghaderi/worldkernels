r"""Trajectory prefix cache.

A radix tree (trie) keyed by action-history hashes. Two sessions that issue
the same sequence of actions from the same initial state produce the same
latents; the cache lets the second session reuse the first's latent blocks for
the shared prefix instead of recomputing them.

Block ids stored at each node reference blocks in a
`BlockManager`; this module
holds only the index, not tensors, so it is torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TrajectoryCache"]


@dataclass
class _TrieNode:
    block_ids: list[int] = field(default_factory=list)
    children: dict[str, "_TrieNode"] = field(default_factory=dict)


@dataclass
class PrefixMatch:
    r"""Result of a trajectory-prefix lookup.

    Args:
        length: Number of leading actions matched.
        block_ids: Cached latent block ids for the matched prefix.
    """

    length: int
    block_ids: list[int]


class TrajectoryCache:
    r"""Radix-tree cache mapping action-history prefixes to latent block ids."""

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    @property
    def size(self) -> int:
        r"""Number of cached trajectory steps."""
        return self._size

    def insert(self, action_hashes: list[str], block_ids: list[int]) -> None:
        r"""Cache the per-step latent block id for an action sequence.

        Args:
            action_hashes: Content hashes of each action in order.
            block_ids: Latent block id produced after each action; same length.
        """
        if len(action_hashes) != len(block_ids):
            raise ValueError("action_hashes and block_ids must have equal length")
        node = self._root
        for action_hash, block_id in zip(action_hashes, block_ids):
            child = node.children.get(action_hash)
            if child is None:
                child = _TrieNode()
                node.children[action_hash] = child
                self._size += 1
            child.block_ids = [block_id]
            node = child

    def match(self, action_hashes: list[str]) -> PrefixMatch:
        r"""Return the longest cached prefix of ``action_hashes``."""
        node = self._root
        matched: list[int] = []
        for action_hash in action_hashes:
            child = node.children.get(action_hash)
            if child is None or not child.block_ids:
                break
            matched.append(child.block_ids[0])
            node = child
        return PrefixMatch(length=len(matched), block_ids=matched)

    def clear(self) -> None:
        self._root = _TrieNode()
        self._size = 0
