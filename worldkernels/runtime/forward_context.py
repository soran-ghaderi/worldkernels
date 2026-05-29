r"""Thread-local forward context.

Threads execution state — attention metadata, parallel-state degrees, the
active cache backend, the current CUDA stream — through a model forward pass
without passing it as kwargs through every layer. Components read it via
`get_forward_context()`; the runner sets it via `set_forward_context()`.

Ported in spirit from ``vllm_omni/diffusion/forward_context.py``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["ForwardContext", "get_forward_context", "set_forward_context"]


@dataclass
class ForwardContext:
    r"""State threaded through one batched forward pass.

    Args:
        attn_metadata: Backend-specific attention state (KV block tables,
            sequence lengths) for the current batch.
        cache_backend: Active denoise-step cache (e.g. TeaCache), or ``None``.
        pool: Active per-runner buffer pool (`LatentPool`), or ``None``.
        attention_backend: Resolved attention backend name (``"flash"`` /
            ``"sdpa"``) when the runtime forces a choice; ``None`` lets the
            selector fall through to the platform default.
        stream: Current compute CUDA stream, or ``None`` on CPU.
        sp_world_size: Sequence-parallel world size for the active forward.
        sp_padding: Sequence padding added to make the length SP-divisible.
        extra: Backend escape hatch for additional per-forward state.
    """

    attn_metadata: Any = None
    cache_backend: Any = None
    pool: Any = None
    attention_backend: str | None = None
    iteration_batching: bool = True
    stream: Any = None
    sp_world_size: int = 1
    sp_padding: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_local = threading.local()


def get_forward_context() -> ForwardContext:
    r"""Return the active context, raising if none is set."""
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        raise RuntimeError(
            "no active ForwardContext; wrap model execution in set_forward_context()"
        )
    return ctx


@contextmanager
def set_forward_context(ctx: ForwardContext) -> Iterator[ForwardContext]:
    r"""Bind ``ctx`` as the active forward context for the calling thread."""
    previous = getattr(_local, "ctx", None)
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = previous
