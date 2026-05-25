r"""Compatibility-group batching.

Two step requests may share a batched forward pass only if they run the same
world model in the same regime. `CompatibilityKey` is the equivalence
class; `group_requests()` partitions a queue into batchable groups, each
capped at the scheduler's ``max_batch_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.core.request import StepRequest

__all__ = ["CompatibilityKey", "CompatibilityGroup", "group_requests"]


@dataclass(frozen=True)
class CompatibilityKey:
    r"""Equivalence class for batching: same world instance, same regime."""

    world_id: int
    transition_mode: str
    decode: bool

    @classmethod
    def of(cls, request: "StepRequest") -> "CompatibilityKey":
        return cls(
            world_id=id(request.world),
            transition_mode=request.world.transition_mode.value,
            decode=request.decode,
        )


@dataclass
class CompatibilityGroup:
    r"""A set of step requests that can share one batched forward pass."""

    key: CompatibilityKey
    requests: list["StepRequest"]

    @property
    def size(self) -> int:
        return len(self.requests)


def group_requests(
    requests: list["StepRequest"],
    *,
    max_batch_size: int,
) -> list[CompatibilityGroup]:
    r"""Partition ``requests`` into compatibility groups capped at ``max_batch_size``.

    Insertion order is preserved within each group; a group that exceeds the
    cap is split into consecutive batches.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    by_key: dict[CompatibilityKey, list[StepRequest]] = {}
    for request in requests:
        by_key.setdefault(CompatibilityKey.of(request), []).append(request)

    groups: list[CompatibilityGroup] = []
    for key, members in by_key.items():
        for start in range(0, len(members), max_batch_size):
            groups.append(CompatibilityGroup(key, members[start : start + max_batch_size]))
    return groups
