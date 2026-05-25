r"""Root pytest configuration: shared fixtures and global safety guards."""

from __future__ import annotations

import os

import pytest
import torch

os.environ["WORLDKERNELS_NO_AUTO_INSTALL"] = "1"

from tests._helpers.factories import make_world_config  # noqa: E402
from tests._helpers.mocks import MockWorld  # noqa: E402
from worldkernels.core.engine import WorldKernel  # noqa: E402
from worldkernels.worlds import (
    hub as _hub,  # noqa: E402
    registry as _registry,  # noqa: E402
)


@pytest.fixture(autouse=True)
def _no_pip_install(monkeypatch):
    r"""Hard guard: under no circumstances may a test trigger pip install."""
    monkeypatch.setattr(_hub, "ensure_model_deps", lambda model_id: None)


@pytest.fixture
def torch_seed():
    torch.manual_seed(0xABCDEF)
    yield 0xABCDEF


@pytest.fixture
def mock_world() -> MockWorld:
    w = MockWorld()
    w.initialize(device="cpu", dtype=torch.float32)
    return w


@pytest.fixture
def world_config():
    return make_world_config()


@pytest.fixture
def engine():
    wk = WorldKernel(device="cpu", max_sessions=4)
    wk.load_model("dummy")
    yield wk
    wk.shutdown()


@pytest.fixture
def session(engine, world_config):
    return engine.create_session("dummy", config=world_config, seed=42)


@pytest.fixture
def registered_mock_world():
    name = "_pytest_mock_world"
    _registry.register_world(name, MockWorld)
    try:
        yield name
    finally:
        _registry._REGISTRY.pop(name, None)
