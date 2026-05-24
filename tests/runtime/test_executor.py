r"""Tests for worldkernels/runtime/executor.py."""

from __future__ import annotations

import torch

from tests._helpers.factories import make_action, make_latent_state
from tests._helpers.mocks import SlowMockWorld
from worldkernels.runtime.connectors import LocalConnector
from worldkernels.runtime.executor import Executor
from worldkernels.runtime.stages import (
    StageConfig,
    StageOutput,
    StageType,
)


class TestExecutorInit:
    def test_defaults(self):
        ex = Executor(device="cpu", dtype=torch.float32)
        assert ex.device == "cpu"
        assert ex.dtype == torch.float32
        assert StageType.ENCODE in ex.stage_configs
        assert isinstance(ex.connector, LocalConnector)

    def test_custom_stage_configs(self):
        cfgs = (StageConfig(StageType.DECODE, enabled=False),)
        ex = Executor(device="cpu", dtype=torch.float32, stage_configs=cfgs)
        assert ex.stage_configs[StageType.DECODE].enabled is False

    def test_custom_connector(self):
        conn = LocalConnector()
        ex = Executor(device="cpu", dtype=torch.float32, connector=conn)
        assert ex.connector is conn


class TestStageEnabled:
    def test_unknown_stage_treated_as_enabled(self):
        ex = Executor(device="cpu", dtype=torch.float32, stage_configs=())
        assert ex._stage_enabled(StageType.ENCODE) is True

    def test_disabled_stage(self):
        ex = Executor(
            device="cpu",
            dtype=torch.float32,
            stage_configs=(StageConfig(StageType.DECODE, enabled=False),),
        )
        assert ex._stage_enabled(StageType.DECODE) is False


class TestPerStageExecution:
    def test_execute_encode_returns_stage_output(self, mock_world):
        ex = Executor(device="cpu", dtype=torch.float32)
        out = ex.execute_encode(mock_world, make_action("k"))
        assert isinstance(out, StageOutput)
        assert out.stage_type == StageType.ENCODE
        assert torch.equal(out.data, torch.tensor([1.0, 2.0, 3.0]))
        assert out.timing_ms >= 0

    def test_execute_transition_records_mode(self, mock_world):
        ex = Executor(device="cpu", dtype=torch.float32)
        state = make_latent_state(shape=(1,), seed=0)
        out = ex.execute_transition(mock_world, state, torch.zeros(3))
        assert out.stage_type == StageType.TRANSITION
        assert out.metadata["transition_mode"] == mock_world.transition_mode.value

    def test_execute_decode_returns_observation(self, mock_world):
        ex = Executor(device="cpu", dtype=torch.float32)
        state = make_latent_state(shape=(1,), seed=0)
        out = ex.execute_decode(mock_world, state, ["frames"])
        assert out.stage_type == StageType.DECODE
        assert out.data.frames is not None


class TestUnifiedStep:
    def test_runs_all_three_stages(self, mock_world):
        ex = Executor(device="cpu", dtype=torch.float32)
        state = make_latent_state(shape=(1,), seed=0)
        new_state, obs = ex.step(
            world=mock_world,
            state=state,
            action=make_action("null"),
            modalities=["frames"],
            step_index=7,
        )
        assert obs.step_index == 7
        assert obs.generation_time_ms >= 0
        assert obs.stage_timing is not None
        assert "encode_action:null" in mock_world.calls
        assert "transition" in mock_world.calls
        assert any(c.startswith("decode:") for c in mock_world.calls)
        assert new_state.data is not None

    def test_decode_false_returns_blank_observation(self, mock_world):
        ex = Executor(device="cpu", dtype=torch.float32)
        state = make_latent_state(shape=(1,), seed=0)
        _, obs = ex.step(
            world=mock_world,
            state=state,
            action=make_action("null"),
            modalities=["frames"],
            step_index=3,
            decode=False,
        )
        assert obs.decode_skipped is True
        assert obs.step_index == 3
        assert obs.frames is None

    def test_decode_disabled_stage_also_skips(self, mock_world):
        ex = Executor(
            device="cpu",
            dtype=torch.float32,
            stage_configs=(StageConfig(StageType.DECODE, enabled=False),),
        )
        state = make_latent_state(shape=(1,), seed=0)
        _, obs = ex.step(
            world=mock_world,
            state=state,
            action=make_action("null"),
            modalities=["frames"],
            step_index=0,
        )
        assert obs.decode_skipped is True

    def test_timing_breakdown_non_negative(self):
        ex = Executor(device="cpu", dtype=torch.float32)
        world = SlowMockWorld()
        world.initialize("cpu", torch.float32)
        state = make_latent_state(shape=(1,), seed=0)
        _, obs = ex.step(
            world=world,
            state=state,
            action=make_action("null"),
            modalities=["frames"],
            step_index=0,
        )
        t = obs.stage_timing
        assert t.encode_action_ms > 0
        assert t.transition_ms > 0
        assert t.decode_observation_ms > 0
        assert t.total_ms <= obs.generation_time_ms + 1e-3
