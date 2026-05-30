r"""WorldRunner — the compute layer.

Holds the executor and runs batched step requests under a `ForwardContext`.
Owns the runtime-feature instances (latent pool, denoise-step cache, paged KV
cache + block allocator) gated by `RuntimeConfig`; off-flags fall through to
the eager path with identical numerics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from worldkernels.runtime.executor import Executor
from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

if TYPE_CHECKING:
    import torch

    from worldkernels.config import RuntimeConfig
    from worldkernels.core.observation import Observation
    from worldkernels.core.request import StepRequest
    from worldkernels.core.session import LatentState
    from worldkernels.runtime.cache.teacache import TeaCache
    from worldkernels.runtime.memory.block_manager import BlockManager
    from worldkernels.runtime.memory.kv_cache import KVCacheManager
    from worldkernels.runtime.memory.latent_pool import LatentPool

__all__ = ["WorldRunner"]


class WorldRunner:
    r"""Runs world-model steps on one device.

    Args:
        device: Target device string.
        dtype: Compute dtype.
        config: Runtime config (component toggles).
    """

    def __init__(
        self,
        device: str,
        dtype: "torch.dtype",
        config: "RuntimeConfig | None" = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.config = config
        self.executor = Executor(device=device, dtype=dtype, config=config)

        self.pool: "LatentPool | None" = None
        self.teacache: "TeaCache | None" = None
        self.block_manager: "BlockManager | None" = None
        self.kv_cache: "KVCacheManager | None" = None

        if config is not None and config.latent_pool:
            from worldkernels.runtime.memory.latent_pool import LatentPool

            self.pool = LatentPool(device=device)

        if config is not None and config.teacache:
            from worldkernels.runtime.cache.teacache import TeaCache

            self.teacache = TeaCache()

    def ensure_kv_cache(self, frames_per_block: int = 16, num_blocks: int = 64) -> None:
        r"""Allocate the paged KV substrate on first causal-world load.

        No-op when ``kv_cache_paged`` is off or already created.
        """
        if self.kv_cache is not None:
            return
        if self.config is None or not self.config.kv_cache_paged:
            return
        from worldkernels.runtime.memory.block_manager import BlockManager
        from worldkernels.runtime.memory.kv_cache import KVCacheManager

        self.block_manager = BlockManager(
            block_shape=(frames_per_block,),
            num_blocks=num_blocks,
            device=self.device,
            dtype=self.dtype,
        )
        self.kv_cache = KVCacheManager(self.block_manager)

    def run_batch(
        self,
        requests: list["StepRequest"],
    ) -> list[tuple["LatentState", "Observation"]]:
        r"""Execute a batch of step requests under a forward context.

        Per-session overrides on the first request (the safe subset) are
        overlaid onto the runner defaults for the duration of the batch.
        """
        overrides = (requests[0].overrides or {}) if requests else {}

        teacache_on = overrides.get("teacache", True)
        cache_backend = self.teacache if teacache_on else None

        attn = overrides.get("attention_backend") or self._default_attention_backend()
        attn = None if attn == "auto" else attn

        iter_batch = overrides.get("iteration_batching")
        if iter_batch is None:
            iter_batch = bool(self.config and self.config.iteration_batching)

        ctx = ForwardContext(
            cache_backend=cache_backend,
            pool=self.pool,
            attention_backend=attn,
            iteration_batching=iter_batch,
        )
        with set_forward_context(ctx):
            return self.executor.execute_batched(requests)

    def _default_attention_backend(self) -> str | None:
        if self.config is None:
            return None
        return self.config.attention_backend
