r"""Tests for worldkernels/worlds/pipelines/cosmos_predict2/__init__.py (lazy exports)."""

from __future__ import annotations

import importlib

import pytest


def test_lazy_exports():
    mod = importlib.import_module("worldkernels.models.cosmos_predict2")
    from worldkernels.models.cosmos_predict2.deps import ensure_cosmos_predict2
    from worldkernels.models.cosmos_predict2.pipeline import (
        CosmosPredict2Latent,
        CosmosPredict2Pipeline,
    )

    assert mod.CosmosPredict2Latent is CosmosPredict2Latent
    assert mod.CosmosPredict2Pipeline is CosmosPredict2Pipeline
    assert mod.ensure_cosmos_predict2 is ensure_cosmos_predict2


def test_unknown_attribute_raises():
    mod = importlib.import_module("worldkernels.models.cosmos_predict2")
    with pytest.raises(AttributeError):
        mod.nope
