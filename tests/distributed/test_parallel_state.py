r"""Single-rank identity tests for the distributed substrate.

Multi-rank correctness (process groups, collectives) is exercised by a
torchrun launch; here we verify that the all-ones config is a pure no-op.
"""

from __future__ import annotations

import pytest
import torch

from worldkernels.config import ParallelConfig
from worldkernels.distributed import (
    CFGParallelMixin,
    get_cfg_parallel_world_size,
    get_sequence_parallel_world_size,
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    init_distributed,
    is_initialized,
    ring_rotate,
    tensor_parallel_all_gather,
    tensor_parallel_all_reduce,
    ulysses_all_to_all,
)


@pytest.fixture(autouse=True)
def _single_rank():
    init_distributed(ParallelConfig())
    yield


class TestParallelState:
    def test_single_rank_initialized(self):
        assert is_initialized()
        assert get_tensor_parallel_rank() == 0
        assert get_tensor_parallel_world_size() == 1
        assert get_sequence_parallel_world_size() == 1
        assert get_cfg_parallel_world_size() == 1
        assert get_tensor_parallel_group() is None

    def test_multi_rank_world_size_product(self):
        cfg = ParallelConfig(tensor_parallel_size=2, ulysses_degree=2, cfg_parallel_size=2)
        assert cfg.world_size == 8
        assert cfg.sequence_parallel_size == 2

    def test_cfg_parallel_size_validated(self):
        with pytest.raises(ValueError, match="cfg_parallel_size"):
            ParallelConfig(cfg_parallel_size=3)


class TestCollectivesSingleRank:
    def test_all_reduce_is_identity(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        assert tensor_parallel_all_reduce(t) is t

    def test_all_gather_is_identity(self):
        t = torch.tensor([1.0, 2.0])
        assert tensor_parallel_all_gather(t) is t

    def test_ulysses_all_to_all_is_identity(self):
        x = torch.rand(2, 4, 8)
        assert ulysses_all_to_all(x, scatter_dim=1, gather_dim=2) is x

    def test_ring_rotate_is_identity(self):
        t = torch.rand(3, 5)
        assert ring_rotate(t) is t


class TestCFGParallelSingleRank:
    def test_not_enabled(self):
        assert CFGParallelMixin.cfg_parallel_enabled() is False

    def test_forward_combines_cond_and_uncond(self):
        def model(*, value):
            return torch.full((2,), float(value))

        out = CFGParallelMixin.cfg_parallel_forward(
            model,
            cond_kwargs={"value": 3.0},
            uncond_kwargs={"value": 1.0},
            guidance_scale=2.0,
        )
        assert torch.allclose(out, torch.full((2,), 1.0 + 2.0 * (3.0 - 1.0)))
