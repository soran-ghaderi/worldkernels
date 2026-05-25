r"""Tests for the worldkernels/runtime/memory subsystem (CPU-safe)."""

from __future__ import annotations

import pytest
import torch

from worldkernels.runtime.memory import (
    BlockManager,
    KVCacheManager,
    LatentPool,
    MemoryTier,
    Offloader,
    TrajectoryCache,
)


def _block_manager(num_blocks: int = 4) -> BlockManager:
    return BlockManager(
        block_shape=(4, 8), num_blocks=num_blocks, device="cpu", dtype=torch.float32
    )


class TestBlockManager:
    def test_allocate_and_free(self):
        bm = _block_manager(4)
        ids = bm.allocate(3)
        assert len(ids) == 3
        assert bm.num_allocated == 3
        assert bm.num_free == 1
        bm.free(ids)
        assert bm.num_free == 4

    def test_block_view_shape(self):
        bm = _block_manager()
        (block_id,) = bm.allocate(1)
        assert bm.block(block_id).shape == (4, 8)

    def test_exhaustion_raises(self):
        bm = _block_manager(2)
        bm.allocate(2)
        with pytest.raises(MemoryError, match="exhausted"):
            bm.allocate(1)

    def test_copy_on_write_share(self):
        bm = _block_manager()
        ids = bm.allocate(1)
        bm.share(ids)
        assert bm.refcount(ids[0]) == 2
        bm.free(ids)
        assert bm.num_allocated == 1
        bm.free(ids)
        assert bm.num_allocated == 0

    def test_reset(self):
        bm = _block_manager()
        bm.allocate(3)
        bm.reset()
        assert bm.num_free == 4


class TestLatentPool:
    def test_acquire_returns_requested_shape(self):
        pool = LatentPool(device="cpu")
        buf = pool.acquire((2, 3), torch.float32)
        assert buf.shape == (2, 3)
        assert pool.num_in_use == 1

    def test_release_recycles_buffer(self):
        pool = LatentPool(device="cpu")
        buf = pool.acquire((2, 3), torch.float32)
        pool.release(buf)
        assert pool.num_pooled == 1
        assert pool.acquire((2, 3), torch.float32) is buf

    def test_distinct_shapes_not_mixed(self):
        pool = LatentPool(device="cpu")
        pool.release(pool.acquire((2, 3), torch.float32))
        other = pool.acquire((4, 4), torch.float32)
        assert other.shape == (4, 4)


class TestOffloader:
    def test_tier_of_cpu_tensor(self):
        assert Offloader.tier_of(torch.zeros(2)) is MemoryTier.HOST

    def test_offload_to_host_is_noop_on_cpu(self):
        off = Offloader(device="cpu")
        t = torch.zeros(2)
        assert off.offload(t, MemoryTier.HOST) is t

    def test_nvme_tier_not_implemented(self):
        off = Offloader(device="cpu")
        with pytest.raises(NotImplementedError):
            off.offload(torch.zeros(2), MemoryTier.NVME)


class TestKVCacheManager:
    def test_allocate_and_block_table(self):
        kv = KVCacheManager(_block_manager(8))
        kv.allocate("s1", 3)
        assert len(kv.block_table("s1")) == 3

    def test_append_grows_table(self):
        kv = KVCacheManager(_block_manager(8))
        kv.allocate("s1", 2)
        kv.append("s1", 2)
        assert len(kv.block_table("s1")) == 4

    def test_fork_shares_blocks_copy_on_write(self):
        bm = _block_manager(8)
        kv = KVCacheManager(bm)
        kv.allocate("s1", 3)
        kv.fork("s1", "s2")
        assert kv.block_table("s2") == kv.block_table("s1")
        assert bm.refcount(kv.block_table("s1")[0]) == 2
        kv.free("s1")
        assert bm.num_allocated == 3

    def test_free_releases_blocks(self):
        bm = _block_manager(8)
        kv = KVCacheManager(bm)
        kv.allocate("s1", 3)
        kv.free("s1")
        assert bm.num_allocated == 0


class TestTrajectoryCache:
    def test_match_full_prefix(self):
        cache = TrajectoryCache()
        cache.insert(["a", "b", "c"], [10, 11, 12])
        match = cache.match(["a", "b", "c"])
        assert match.length == 3
        assert match.block_ids == [10, 11, 12]

    def test_match_partial_prefix(self):
        cache = TrajectoryCache()
        cache.insert(["a", "b"], [10, 11])
        assert cache.match(["a", "b", "x"]).length == 2

    def test_match_no_prefix(self):
        cache = TrajectoryCache()
        cache.insert(["a"], [10])
        assert cache.match(["z"]).length == 0

    def test_insert_length_mismatch_raises(self):
        cache = TrajectoryCache()
        with pytest.raises(ValueError, match="equal length"):
            cache.insert(["a", "b"], [1])
