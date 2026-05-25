r"""Tests for worldkernels/engine/world_engine.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tests._helpers.factories import make_world_config
from tests._helpers.mocks import MockWorld, register_mock_world, unregister_world
from worldkernels.core.errors import (
    SessionLimitError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldNotFoundError,
)
from worldkernels.engine import WorldEngine
from worldkernels.engine.world_engine import _default_dtype


class TestDefaultDtype:
    def test_cpu_is_float32(self):
        assert _default_dtype("cpu") == torch.float32

    def test_cuda_without_gpu_falls_back_float32(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert _default_dtype("cuda") == torch.float32

    def test_cuda_ampere_uses_bfloat16(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            assert _default_dtype("cuda") == torch.bfloat16

    def test_cuda_pre_ampere_uses_float16(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            assert _default_dtype("cuda") == torch.float16


class TestWorldKernelInit:
    def test_defaults(self):
        wk = WorldEngine(device="cpu")
        try:
            assert wk.device == "cpu"
            assert wk.max_sessions == 4
            assert wk.offload_idle is True
            assert wk.dtype == torch.float32
            assert wk._scheduler is not None
            assert wk.list_worlds() == []
            assert wk.list_sessions() == []
        finally:
            wk.shutdown()

    def test_overrides(self):
        wk = WorldEngine(device="cpu", max_sessions=8, offload_idle=False)
        try:
            assert wk.max_sessions == 8
            assert wk.offload_idle is False
        finally:
            wk.shutdown()


class TestLoadModel:
    def test_load_dummy(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            assert "dummy" in wk.list_worlds()
        finally:
            wk.shutdown()

    def test_load_unknown_raises(self):
        wk = WorldEngine(device="cpu")
        try:
            with pytest.raises(WorldNotFoundError):
                wk.load_model("nonexistent_xyz")
        finally:
            wk.shutdown()

    def test_load_with_alias(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy", alias="d1")
            assert "d1" in wk.list_worlds()
            assert "dummy" not in wk.list_worlds()
        finally:
            wk.shutdown()

    def test_already_loaded_raises(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            with pytest.raises(WorldAlreadyLoadedError):
                wk.load_model("dummy")
        finally:
            wk.shutdown()

    def test_hf_repo_id_resolved_via_hub(self):
        wk = WorldEngine(device="cpu")
        try:
            register_mock_world("dreamdojo")
            try:
                wk.load_model("nvidia/DreamDojo", alias="dd")
                assert "dd" in wk.list_worlds()
            finally:
                unregister_world("dreamdojo")
        finally:
            wk.shutdown()

    def test_init_failure_raises_world_init_error(self):
        class BadWorld(MockWorld):
            def initialize(self, device, dtype):
                raise RuntimeError("boom")

        register_mock_world("_bad", BadWorld)
        wk = WorldEngine(device="cpu")
        try:
            with pytest.raises(WorldInitError, match="boom"):
                wk.load_model("_bad")
        finally:
            unregister_world("_bad")
            wk.shutdown()


class TestCreateSession:
    def test_creates_active(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy", config=make_world_config())
            assert s.world_id == "dummy"
            assert s.session_id in wk._sessions
            assert s._world is wk._worlds["dummy"]
            assert s._scheduler is wk._scheduler
        finally:
            wk.shutdown()

    def test_default_config(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy")
            assert s.config is not None
            assert s.config.height == 480
        finally:
            wk.shutdown()

    def test_unknown_world_raises(self):
        wk = WorldEngine(device="cpu")
        try:
            with pytest.raises(WorldNotFoundError):
                wk.create_session("nope")
        finally:
            wk.shutdown()

    def test_session_limit(self):
        wk = WorldEngine(device="cpu", max_sessions=2)
        try:
            wk.load_model("dummy")
            cfg = make_world_config()
            wk.create_session("dummy", config=cfg)
            wk.create_session("dummy", config=cfg)
            with pytest.raises(SessionLimitError):
                wk.create_session("dummy", config=cfg)
        finally:
            wk.shutdown()

    def test_default_seed_is_zero(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy")
            assert s.seed == 0
        finally:
            wk.shutdown()

    def test_vram_check_raises(self):
        wk = WorldEngine(device="cpu")
        try:
            hungry = _HungryWorld()
            hungry.initialize("cpu", torch.float32)
            wk._worlds["_hungry"] = hungry
            wk.device = "cuda:0"
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.mem_get_info", return_value=(1024 * 1024, 1024 * 1024)),
            ):
                with pytest.raises(VRAMExhaustedError):
                    wk.create_session("_hungry", config=make_world_config())
        finally:
            wk.shutdown()

    def test_vram_check_passes_when_enough_free(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            wk.device = "cuda:0"
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.mem_get_info", return_value=(10 * 1024**3, 10 * 1024**3)),
            ):
                s = wk.create_session("dummy", config=make_world_config())
                assert s is not None
        finally:
            wk.shutdown()


class _HungryWorld(MockWorld):
    name = "hungry"

    def profile_vram(self, config):
        return 1e9


class TestSessionManagement:
    def test_get_session_returns_none_for_unknown(self):
        wk = WorldEngine(device="cpu")
        try:
            assert wk.get_session("missing") is None
        finally:
            wk.shutdown()

    def test_get_session_returns_session(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy")
            assert wk.get_session(s.session_id) is s
        finally:
            wk.shutdown()

    def test_close_session_terminates(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy")
            wk.close_session(s.session_id)
            assert s.session_id not in wk._sessions
            from worldkernels.core.session import SessionStatus

            assert s.status == SessionStatus.TERMINATED
        finally:
            wk.shutdown()

    def test_close_session_unknown_is_noop(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.close_session("missing")
        finally:
            wk.shutdown()

    def test_list_sessions(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s1 = wk.create_session("dummy")
            s2 = wk.create_session("dummy")
            sids = set(wk.list_sessions())
            assert sids == {s1.session_id, s2.session_id}
        finally:
            wk.shutdown()


class TestUnloadModel:
    def test_unloads_and_closes_sessions(self):
        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            s = wk.create_session("dummy")
            wk.unload_model("dummy")
            assert "dummy" not in wk.list_worlds()
            assert s.session_id not in wk._sessions
        finally:
            wk.shutdown()

    def test_unload_unknown_raises(self):
        wk = WorldEngine(device="cpu")
        try:
            with pytest.raises(WorldNotFoundError):
                wk.unload_model("ghost")
        finally:
            wk.shutdown()


class TestShutdown:
    def test_shutdown_clears_state(self):
        wk = WorldEngine(device="cpu")
        wk.load_model("dummy")
        wk.create_session("dummy")
        wk.shutdown()
        assert wk._worlds == {}
        assert wk._sessions == {}
