r"""Root pytest configuration: shared fixtures and global safety guards."""

from __future__ import annotations

import os

import pytest
import torch

os.environ["WORLDKERNELS_NO_AUTO_INSTALL"] = "1"
os.environ["WORLDKERNELS_QUIET"] = "1"

from tests._helpers.factories import make_world_config  # noqa: E402
from tests._helpers.mocks import MockWorld  # noqa: E402
from worldkernels.engine import WorldEngine  # noqa: E402
from worldkernels.worlds import (
    hub as _hub,  # noqa: E402
    registry as _registry,  # noqa: E402
)


@pytest.fixture(autouse=True)
def _no_pip_install(monkeypatch):
    r"""Hard guard: under no circumstances may a test trigger pip/uv install, git clone, HF download, or venv materialization."""
    from worldkernels.bootstrap import deps as _deps
    from worldkernels.bootstrap import weights as _weights
    from worldkernels.runtime import envs as _envs

    def _block_pip(card, progress=None, allow_fetch=True, target_python=None):
        return None

    def _block_git(card, progress=None, allow_fetch=True):
        return None

    def _block_install(packages, target_python=None, progress=None, constraints=None):
        return None

    def _block_weights(card, variant=None, ckpt_path=None, progress=None, allow_fetch=True):
        if ckpt_path is not None:
            return ckpt_path
        return None

    def _block_env(model_id, requirements, device="cuda", progress=None, allow_fetch=True):
        raise AssertionError(
            f"test triggered materialize_env({model_id!r}); add a monkeypatch if intended"
        )

    monkeypatch.setattr(_hub, "ensure_model_deps", lambda model_id: None)
    monkeypatch.setattr(_deps, "provision_python_deps", _block_pip)
    monkeypatch.setattr(_deps, "provision_git_packages", _block_git)
    monkeypatch.setattr(_deps, "install_packages", _block_install)
    monkeypatch.setattr(_weights, "provision_weights", _block_weights)
    monkeypatch.setattr(_envs, "materialize_env", _block_env)


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
    wk = WorldEngine(device="cpu", max_sessions=4)
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
