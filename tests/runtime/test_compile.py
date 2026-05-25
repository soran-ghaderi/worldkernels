r"""Tests for the compile backends (CPU-safe paths)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from worldkernels.runtime.compile import CUDAGraphRunner, regionally_compile


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Model(nn.Module):
    _repeated_blocks = ["blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TestRegionallyCompile:
    def test_returns_module_and_preserves_output(self):
        model = _Model()
        x = torch.rand(2, 4)
        expected = model(x)
        compiled = regionally_compile(model)
        assert compiled is model
        assert torch.allclose(compiled(x), expected, atol=1e-5)

    def test_compiles_each_repeated_block(self):
        model = regionally_compile(_Model())
        assert len(model.blocks) == 3


class TestCUDAGraphRunner:
    def test_requires_cuda(self):
        runner = CUDAGraphRunner()
        assert runner.is_captured is False
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA"):
                runner.capture(lambda t: t, torch.zeros(2))

    def test_replay_before_capture_raises(self):
        with pytest.raises(RuntimeError, match="not captured"):
            CUDAGraphRunner().replay(torch.zeros(2))
