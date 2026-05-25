r"""Memory and cache configuration for the runtime."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CacheConfig"]


@dataclass
class CacheConfig:
    r"""Block-based memory and caching policy.

    Args:
        block_frames: Latent frames per fixed-size memory block.
        gpu_memory_fraction: Fraction of free VRAM the block allocator claims.
        enable_kv_paging: Page the KV cache for causal world models.
        enable_trajectory_cache: Share latent blocks across sessions with a
            common action-history prefix (radix-tree prefix cache).
        offload_to_cpu: Allow paused-session state to spill to pinned host memory.
        offload_to_nvme: Allow cold state to spill to an NVMe-backed mmap.
        nvme_path: Directory for NVMe spill files (required if ``offload_to_nvme``).
    """

    block_frames: int = 16
    gpu_memory_fraction: float = 0.8
    enable_kv_paging: bool = True
    enable_trajectory_cache: bool = True
    offload_to_cpu: bool = True
    offload_to_nvme: bool = False
    nvme_path: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError(
                f"gpu_memory_fraction must be in (0, 1], got {self.gpu_memory_fraction}"
            )
        if self.offload_to_nvme and self.nvme_path is None:
            raise ValueError("nvme_path is required when offload_to_nvme is True")
