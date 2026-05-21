r"""Tests for worldkernels/worlds/adapters/cosmos/__init__.py."""

from __future__ import annotations

import importlib

import pytest


def test_lazy_export():
    mod = importlib.import_module("worldkernels.worlds.adapters.cosmos")
    from worldkernels.worlds.adapters.cosmos.adapter import CosmosPredict2World

    assert mod.CosmosPredict2World is CosmosPredict2World


def test_unknown_attribute_raises():
    mod = importlib.import_module("worldkernels.worlds.adapters.cosmos")
    with pytest.raises(AttributeError):
        mod.does_not_exist
