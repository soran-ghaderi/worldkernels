r"""CPU tests for the native DreamDojo world adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.session import LatentState
from worldkernels.core.state import WorldState
from worldkernels.models.dreamdojo.pipeline import DreamDojoPipeline
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.dreamdojo import DreamDojoWorld

H, W = 480, 640


def _make_pipeline_mock() -> MagicMock:
    p = MagicMock(spec=DreamDojoPipeline)
    p.generate_clip.return_value = (
        torch.randn(1, 16, 4, H // 8, W // 8),
        torch.zeros(1, 3, 13, H, W),
    )
    p.to_uint8.side_effect = DreamDojoPipeline.to_uint8
    p.decode_latent.return_value = torch.zeros(1, 3, 13, H, W)
    return p


def _make_world(**kwargs) -> DreamDojoWorld:
    world = DreamDojoWorld(**kwargs)
    world.pipeline = _make_pipeline_mock()
    return world


def _make_state(world: DreamDojoWorld, seed: int = 7) -> LatentState:
    return world.create_initial_state(WorldConfig(), seed=seed)


class TestMetadata:
    def test_name_and_modes(self):
        assert DreamDojoWorld.name == "dreamdojo"
        assert DreamDojoWorld.stage_exec_modes[StageType.ENCODE] == StageExecMode.SINGLE_SHOT
        assert DreamDojoWorld.stage_exec_modes[StageType.TRANSITION] == StageExecMode.ITERATIVE
        assert DreamDojoWorld.transition_mode == TransitionMode.BIDIRECTIONAL
        assert DreamDojoWorld.supports_kv_cache is False
        assert DreamDojoWorld.supports_streaming is False

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="unknown DreamDojo variant"):
            DreamDojoWorld(variant="3b_bad")

    def test_all_variants_use_universal_action_dim(self):
        for variant in ("2b_pretrain", "2b_gr1", "14b_yam"):
            assert DreamDojoWorld(variant=variant).action_dim == 384


class TestGeometry:
    def test_snaps_generic_defaults_to_native(self):
        world = _make_world()
        assert world._geometry(WorldConfig()) == (480, 640, 12)

    def test_explicit_values_honored(self):
        world = _make_world()
        cfg = WorldConfig(height=240, width=320, frames_per_step=4)
        assert world._geometry(cfg) == (240, 320, 4)


class TestEncodeAction:
    def test_null_action(self):
        world = _make_world()
        out = world.encode_action(Action("null", {}))
        assert out.shape == (1, 12, 384)
        assert torch.all(out == 0)

    def test_broadcast_1d(self):
        world = _make_world()
        out = world.encode_action(Action("continuous", {"joints": [0.5] * 384}))
        assert out.shape == (1, 12, 384)
        assert torch.all(out == 0.5)

    def test_pad_short_chunk(self):
        world = _make_world()
        joints = [[1.0] * 384] * 5
        out = world.encode_action(Action("continuous", {"joints": joints}))
        assert out.shape == (1, 12, 384)
        assert torch.all(out[0, :5] == 1.0)
        assert torch.all(out[0, 5:] == 0.0)

    def test_truncate_long_chunk(self):
        world = _make_world()
        joints = [[1.0] * 384] * 20
        out = world.encode_action(Action("continuous", {"joints": joints}))
        assert out.shape == (1, 12, 384)


class TestTransition:
    def test_deterministic_seed_advances_with_step(self):
        world = _make_world()
        state = _make_state(world, seed=100)
        action = torch.randn(1, 12, 384)

        new_state = world.transition(state, action)
        assert world.pipeline.generate_clip.call_args.kwargs["seed"] == 100
        ws: WorldState = new_state.data
        assert ws.step_index == 1

        world.transition(new_state, action)
        assert world.pipeline.generate_clip.call_args.kwargs["seed"] == 101

    def test_carries_last_frame_and_video(self):
        world = _make_world()
        state = _make_state(world)
        new_state = world.transition(state, torch.randn(1, 12, 384))
        ws: WorldState = new_state.data
        assert ws.conditioning.image_cond is not None
        assert ws.conditioning.image_cond.dtype == torch.uint8
        assert ws.conditioning.image_cond.shape == (1, 3, H, W)
        assert ws.extras["video"].shape == (1, 3, 13, H, W)

    def test_empty_action_becomes_zero_chunk(self):
        world = _make_world()
        state = _make_state(world)
        world.transition(state, torch.empty(0))
        action = world.pipeline.generate_clip.call_args.args[1]
        assert action.shape == (1, 12, 384)
        assert torch.all(action == 0)

    def test_without_pipeline_raises(self):
        world = DreamDojoWorld()
        with pytest.raises(AssertionError):
            world.transition(LatentState(data=None, device="cpu"), torch.zeros(1, 12, 384))


class TestDecodeObservation:
    def test_frames_from_cached_video(self):
        world = _make_world()
        state = _make_state(world)
        state = world.transition(state, torch.randn(1, 12, 384))
        obs = world.decode_observation(state, ["frames"])
        assert obs.frames is not None and len(obs.frames) == 13
        assert len(obs.frames[0]) == H * W * 3
        world.pipeline.decode_latent.assert_not_called()

    def test_frames_decode_fallback_without_cache(self):
        world = _make_world()
        state = _make_state(world)
        obs = world.decode_observation(state, ["frames"])
        assert obs.frames is not None
        world.pipeline.decode_latent.assert_called_once()

    def test_latent_modality(self):
        world = _make_world()
        state = _make_state(world)
        obs = world.decode_observation(state, ["latent"])
        assert obs.latent is not None
        assert obs.frames is None


class TestCreateInitialState:
    def test_zero_frame_without_image(self):
        world = _make_world()
        state = _make_state(world, seed=3)
        ws: WorldState = state.data
        assert ws.conditioning.image_cond is not None
        assert torch.all(ws.conditioning.image_cond == 0)
        assert ws.extras["seed"].item() == 3
        assert ws.meta.height == 480 and ws.meta.width == 640
        assert ws.step_index == 0

    def test_initial_image_array(self):
        world = _make_world()
        img = torch.rand(3, 480, 640)
        state = world.create_initial_state(WorldConfig(initial_image=img), seed=0)
        ws: WorldState = state.data
        assert ws.conditioning.image_cond.dtype == torch.uint8
        assert not torch.all(ws.conditioning.image_cond == 0)

    def test_without_pipeline_raises(self):
        world = DreamDojoWorld()
        with pytest.raises(AssertionError):
            world.create_initial_state(WorldConfig(), seed=0)


class TestInitialize:
    def test_builds_native_pipeline(self):
        world = DreamDojoWorld(variant="2b_gr1", num_inference_steps=10, guidance_scale=2.0)
        with patch("worldkernels.worlds.dreamdojo.DreamDojoPipeline") as cls:
            world.initialize("cuda", torch.bfloat16)
        cls.assert_called_once_with("2b_gr1", num_steps=10, guidance=2.0)
        cls.return_value.load.assert_called_once_with("cuda", torch.bfloat16, None)


class TestProfileVram:
    def test_positive_without_load(self):
        world = DreamDojoWorld()
        assert world.profile_vram(WorldConfig()) > 0

    def test_grows_with_size(self):
        small = DreamDojoWorld(variant="2b_gr1").profile_vram(WorldConfig())
        large = DreamDojoWorld(variant="14b_gr1").profile_vram(WorldConfig())
        assert large > small


class TestStudentWorld:
    def _make_student(self):
        from worldkernels.worlds.dreamdojo import DreamDojoStudentWorld

        world = DreamDojoStudentWorld(variant="2b_gr1")
        p = MagicMock(spec=DreamDojoPipeline)
        p.stream_chunk.return_value = (
            torch.randn(1, 16, 4, H // 8, W // 8),
            torch.zeros(1, 3, 12, H, W),
        )
        p.to_uint8.side_effect = DreamDojoPipeline.to_uint8
        world.pipeline = p
        return world

    def test_metadata(self):
        from worldkernels.worlds.dreamdojo import DreamDojoStudentWorld

        assert DreamDojoStudentWorld.name == "dreamdojo_student"
        assert DreamDojoStudentWorld.transition_mode == TransitionMode.CAUSAL
        assert DreamDojoStudentWorld.supports_streaming is True

    def test_first_step_uses_initial_frame_and_fresh_actions(self):
        world = self._make_student()
        state = world.create_initial_state(WorldConfig(), seed=5)
        fresh = torch.randn(1, 12, 384)
        new_state = world.transition(state, fresh)

        args, kwargs = world.pipeline.stream_chunk.call_args
        assert args[0].shape == (1, 3, 1, H, W)
        torch.testing.assert_close(args[1], fresh)
        assert kwargs["seed"] == 5

        ws: WorldState = new_state.data
        assert ws.extras["context_px"].shape == (1, 3, 9, H, W)
        assert ws.extras["action_history"].shape == (1, 8, 384)
        torch.testing.assert_close(ws.extras["action_history"], fresh[:, -8:])

    def test_steady_step_prepends_action_history(self):
        world = self._make_student()
        state = world.create_initial_state(WorldConfig(), seed=5)
        state = world.transition(state, torch.randn(1, 12, 384))
        fresh = torch.randn(1, 12, 384)
        world.transition(state, fresh)

        args, kwargs = world.pipeline.stream_chunk.call_args
        assert args[0].shape == (1, 3, 9, H, W)
        assert args[1].shape == (1, 20, 384)
        torch.testing.assert_close(args[1][:, 8:], fresh)
        assert kwargs["seed"] == 6

    def test_registered(self):
        from worldkernels.worlds.dreamdojo import DreamDojoStudentWorld
        from worldkernels.worlds.registry import get_world_class

        assert get_world_class("dreamdojo_student") is DreamDojoStudentWorld


class TestLatentActionEncoding:
    def test_direct_latent_action_fills_slice(self):
        world = _make_world()
        z = [0.5] * 32
        out = world.encode_action(Action("latent", {"latent_action": z}))
        assert out.shape == (1, 12, 384)
        assert torch.all(out[0, :, 352:384] == 0.5)
        assert torch.all(out[0, :, :352] == 0)

    def test_per_frame_latent_actions_padded(self):
        world = _make_world()
        z = torch.randn(3, 32)
        out = world.encode_action(Action("latent", {"latent_action": z.tolist()}))
        assert out.shape == (1, 12, 384)
        torch.testing.assert_close(out[0, :3, 352:384], z)
        torch.testing.assert_close(out[0, 3:, 352:384], z[-1:].expand(9, -1))

    def test_frames_payload_uses_lam(self):
        world = _make_world()
        lam = MagicMock()
        lam.encode_video.return_value = torch.ones(12, 32)
        world._lam = lam
        frames = (torch.rand(13, 3, 48, 64) * 255).to(torch.uint8)
        out = world.encode_action(Action("latent", {"frames": frames.numpy()}))
        lam.encode_video.assert_called_once()
        assert torch.all(out[0, :, 352:384] == 1.0)
