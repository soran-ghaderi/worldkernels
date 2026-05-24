r"""Tests for worldkernels/runtime/stages.py."""

from __future__ import annotations

from worldkernels.runtime.stages import (
    DEFAULT_PIPELINE_STAGES,
    StageConfig,
    StageExecMode,
    StageOutput,
    StageTiming,
    StageType,
    TransitionMode,
)


class TestEnums:
    def test_stage_type_values(self):
        assert StageType.ENCODE.value == "encode"
        assert StageType.TRANSITION.value == "transition"
        assert StageType.DECODE.value == "decode"
        assert StageType("encode") is StageType.ENCODE

    def test_stage_exec_mode_values(self):
        assert StageExecMode.SINGLE_SHOT.value == "single_shot"
        assert StageExecMode.ITERATIVE.value == "iterative"

    def test_transition_mode_values(self):
        assert TransitionMode.BIDIRECTIONAL.value == "bidirectional"
        assert TransitionMode.CAUSAL.value == "causal"
        assert TransitionMode.HYBRID.value == "hybrid"


class TestStageConfig:
    def test_defaults(self):
        cfg = StageConfig(stage_type=StageType.ENCODE)
        assert cfg.stage_type == StageType.ENCODE
        assert cfg.exec_mode == StageExecMode.SINGLE_SHOT
        assert cfg.device is None
        assert cfg.dtype is None
        assert cfg.memory_fraction == 1.0
        assert cfg.backend == "eager"
        assert cfg.enabled is True

    def test_overrides(self):
        cfg = StageConfig(
            stage_type=StageType.TRANSITION,
            exec_mode=StageExecMode.ITERATIVE,
            device="cuda:0",
            dtype="bf16",
            memory_fraction=0.5,
            backend="compile",
            enabled=False,
        )
        assert cfg.exec_mode == StageExecMode.ITERATIVE
        assert cfg.device == "cuda:0"
        assert cfg.dtype == "bf16"
        assert cfg.memory_fraction == 0.5
        assert cfg.backend == "compile"
        assert cfg.enabled is False


class TestStageOutput:
    def test_minimal(self):
        out = StageOutput(stage_type=StageType.ENCODE, data=42)
        assert out.stage_type == StageType.ENCODE
        assert out.data == 42
        assert out.timing_ms == 0.0
        assert out.metadata == {}

    def test_with_meta(self):
        out = StageOutput(
            stage_type=StageType.TRANSITION,
            data="x",
            timing_ms=3.14,
            metadata={"k": 1},
        )
        assert out.metadata == {"k": 1}
        assert out.timing_ms == 3.14

    def test_metadata_default_independent_per_instance(self):
        a = StageOutput(stage_type=StageType.ENCODE, data=1)
        b = StageOutput(stage_type=StageType.ENCODE, data=2)
        a.metadata["k"] = 1
        assert "k" not in b.metadata


class TestStageTiming:
    def test_defaults_zero(self):
        t = StageTiming()
        assert t.encode_action_ms == 0.0
        assert t.transition_ms == 0.0
        assert t.decode_observation_ms == 0.0
        assert t.total_ms == 0.0

    def test_total_sums_components(self):
        t = StageTiming(encode_action_ms=1.0, transition_ms=2.0, decode_observation_ms=4.0)
        assert t.total_ms == 7.0

    def test_as_dict_includes_total(self):
        t = StageTiming(encode_action_ms=1.0, transition_ms=2.0, decode_observation_ms=4.0)
        d = t.as_dict()
        assert d == {
            "encode_action_ms": 1.0,
            "transition_ms": 2.0,
            "decode_observation_ms": 4.0,
            "total_ms": 7.0,
        }


class TestDefaults:
    def test_default_pipeline_stages_order_and_modes(self):
        types = [cfg.stage_type for cfg in DEFAULT_PIPELINE_STAGES]
        modes = [cfg.exec_mode for cfg in DEFAULT_PIPELINE_STAGES]
        assert types == [StageType.ENCODE, StageType.TRANSITION, StageType.DECODE]
        assert modes == [
            StageExecMode.SINGLE_SHOT,
            StageExecMode.ITERATIVE,
            StageExecMode.SINGLE_SHOT,
        ]
