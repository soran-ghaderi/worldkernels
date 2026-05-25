r"""Tests for the quantization registry."""

from __future__ import annotations

import pytest
import torch.nn as nn

from worldkernels.runtime.quantization import QuantizationRegistry, QuantizationScheme


class TestQuantizationRegistry:
    def test_none_scheme_is_identity(self):
        module = nn.Linear(4, 4)
        assert QuantizationRegistry().apply(module, QuantizationScheme.NONE) is module

    def test_accepts_string_scheme(self):
        module = nn.Linear(4, 4)
        assert QuantizationRegistry().apply(module, "none") is module

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError):
            QuantizationRegistry().apply(nn.Linear(4, 4), "bogus")

    def test_int8_without_torchao_raises_actionable_error(self):
        try:
            import torchao  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="worldkernels"):
                QuantizationRegistry().apply(nn.Linear(4, 4), QuantizationScheme.INT8)

    def test_register_custom_applier(self):
        registry = QuantizationRegistry()
        marker = nn.Linear(2, 2)
        registry.register(QuantizationScheme.INT4, lambda _m: marker)
        assert registry.apply(nn.Linear(4, 4), QuantizationScheme.INT4) is marker
