r"""Structured world-model state.

Replaces the opaque ``LatentState.data: Any`` with a typed container that
supports cross-session batching, block-paged latent storage, and KV caching for
causal world models, while keeping a tensor-only ``extras`` escape hatch for
model-specific state.

``BlockHandle`` and ``KVHandle`` are *handles*: the world state holds a
reference to storage owned by the runtime, not raw tensors. During the
restructure they are pass-through wrappers; the block-paged implementations
land with the memory subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = [
    "BlockHandle",
    "KVHandle",
    "ConditioningBundle",
    "WorldStateMeta",
    "WorldState",
]


class BlockHandle:
    r"""Handle to block-paged latent storage.

    Pass-through wrapper over a single tensor for now; the block-table
    implementation backed by the runtime block allocator replaces the body
    without changing this interface.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: "torch.Tensor") -> None:
        self.tensor = tensor

    def clone(self) -> "BlockHandle":
        return BlockHandle(self.tensor.clone())

    def to(self, device: Any) -> "BlockHandle":
        return BlockHandle(self.tensor.to(device))

    @property
    def nbytes(self) -> int:
        return self.tensor.nelement() * self.tensor.element_size()


class KVHandle:
    r"""Handle to a paged KV cache for causal world models.

    Pass-through wrapper for now; ``None`` on bidirectional worlds, which pay
    no KV-cache cost.
    """

    __slots__ = ("layers",)

    def __init__(self, layers: list["torch.Tensor"] | None = None) -> None:
        self.layers = layers or []

    def clone(self) -> "KVHandle":
        return KVHandle([t.clone() for t in self.layers])

    def to(self, device: Any) -> "KVHandle":
        return KVHandle([t.to(device) for t in self.layers])

    @property
    def nbytes(self) -> int:
        return sum(t.nelement() * t.element_size() for t in self.layers)


@dataclass
class ConditioningBundle:
    r"""Conditioning signals common to video world models.

    Args:
        text_emb: Encoded prompt embedding.
        neg_text_emb: Encoded negative-prompt embedding (for CFG).
        image_cond: Encoded image-conditioning tensor (the rollout's last frame).
    """

    text_emb: "torch.Tensor | None" = None
    neg_text_emb: "torch.Tensor | None" = None
    image_cond: "torch.Tensor | None" = None

    def clone(self) -> "ConditioningBundle":
        return ConditioningBundle(
            text_emb=None if self.text_emb is None else self.text_emb.clone(),
            neg_text_emb=None if self.neg_text_emb is None else self.neg_text_emb.clone(),
            image_cond=None if self.image_cond is None else self.image_cond.clone(),
        )

    def to(self, device: Any) -> "ConditioningBundle":
        return ConditioningBundle(
            text_emb=None if self.text_emb is None else self.text_emb.to(device),
            neg_text_emb=None if self.neg_text_emb is None else self.neg_text_emb.to(device),
            image_cond=None if self.image_cond is None else self.image_cond.to(device),
        )

    @property
    def nbytes(self) -> int:
        total = 0
        for t in (self.text_emb, self.neg_text_emb, self.image_cond):
            if t is not None:
                total += t.nelement() * t.element_size()
        return total


@dataclass
class WorldStateMeta:
    r"""Shape and regime metadata; the scheduler reads it to bucket sessions."""

    height: int
    width: int
    dtype: str
    transition_mode: str
    frames_per_step: int


@dataclass
class WorldState:
    r"""Structured per-session world-model state.

    Args:
        latent: Handle to the current latent (block-paged storage).
        conditioning: Conditioning signals for the next transition.
        meta: Shape and regime metadata.
        kv_cache: Paged KV cache for causal worlds; ``None`` for bidirectional.
        step_index: Monotonic simulation step counter.
        extras: Model-specific tensors that do not fit the typed fields.
    """

    latent: BlockHandle
    conditioning: ConditioningBundle
    meta: WorldStateMeta
    kv_cache: KVHandle | None = None
    step_index: int = 0
    extras: dict[str, "torch.Tensor"] = field(default_factory=dict)

    def clone(self) -> "WorldState":
        r"""Copy-on-write clone for session branching."""
        return WorldState(
            latent=self.latent.clone(),
            conditioning=self.conditioning.clone(),
            meta=self.meta,
            kv_cache=None if self.kv_cache is None else self.kv_cache.clone(),
            step_index=self.step_index,
            extras={k: v.clone() for k, v in self.extras.items()},
        )

    def to(self, device: Any) -> "WorldState":
        r"""Move all tensors to ``device`` (or an offload tier)."""
        return WorldState(
            latent=self.latent.to(device),
            conditioning=self.conditioning.to(device),
            meta=self.meta,
            kv_cache=None if self.kv_cache is None else self.kv_cache.to(device),
            step_index=self.step_index,
            extras={k: v.to(device) for k, v in self.extras.items()},
        )

    @property
    def nbytes(self) -> int:
        total = self.latent.nbytes + self.conditioning.nbytes
        if self.kv_cache is not None:
            total += self.kv_cache.nbytes
        total += sum(v.nelement() * v.element_size() for v in self.extras.values())
        return total
