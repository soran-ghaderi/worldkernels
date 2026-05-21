r"""Tests for worldkernels/core/observation.py."""

from __future__ import annotations

from worldkernels.core.observation import Observation
from worldkernels.runtime.stages import StageTiming


class TestObservationConstruction:
    def test_minimal(self):
        obs = Observation(step_index=0, generation_time_ms=12.3)
        assert obs.step_index == 0
        assert obs.generation_time_ms == 12.3
        assert obs.frames is None
        assert obs.latent is None
        assert obs.audio is None
        assert obs.depth is None
        assert obs.segmentation is None
        assert obs.structured == {}
        assert obs.stage_timing is None
        assert obs.decode_skipped is False

    def test_all_modalities(self):
        timing = StageTiming(encode_action_ms=1.0)
        obs = Observation(
            step_index=5,
            generation_time_ms=100.0,
            frames=[b"f"],
            latent=[1, 2, 3],
            audio=b"a",
            depth=b"d",
            segmentation=b"s",
            structured={"k": "v"},
            stage_timing=timing,
            decode_skipped=True,
        )
        assert obs.frames == [b"f"]
        assert obs.latent == [1, 2, 3]
        assert obs.audio == b"a"
        assert obs.depth == b"d"
        assert obs.segmentation == b"s"
        assert obs.structured == {"k": "v"}
        assert obs.stage_timing is timing
        assert obs.decode_skipped is True

    def test_structured_default_is_independent_per_instance(self):
        a = Observation(step_index=0, generation_time_ms=0.0)
        b = Observation(step_index=1, generation_time_ms=0.0)
        a.structured["x"] = 1
        assert "x" not in b.structured
