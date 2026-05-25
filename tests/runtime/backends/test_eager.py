r"""Tests for worldkernels/runtime/backends/eager.py."""

from __future__ import annotations

import torch

from worldkernels.runtime.backends.eager import EagerBackend


class TestEagerBackend:
    def test_name(self):
        assert EagerBackend.name == "eager"

    def test_init(self):
        b = EagerBackend(device="cpu", dtype=torch.float32)
        assert b.device == "cpu"
        assert b.dtype == torch.float32

    def test_run_invokes_function_and_returns_value(self):
        b = EagerBackend(device="cpu", dtype=torch.float32)
        result = b.run(lambda x, y=0: x + y, 1, y=2)
        assert result == 3

    def test_run_in_no_grad_context(self):
        b = EagerBackend(device="cpu", dtype=torch.float32)
        x = torch.ones(2, requires_grad=True)

        def fn(t):
            assert torch.is_grad_enabled() is False
            return t * 2

        out = b.run(fn, x)
        assert out.requires_grad is False
