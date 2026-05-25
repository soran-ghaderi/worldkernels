r"""CPU tests for the Wan image-to-video pipeline and its latent container."""

from __future__ import annotations

import pytest
import torch

from worldkernels.models.wan import WanI2VPipeline, WanLatent


def _latent() -> WanLatent:
    return WanLatent(
        last_frame=torch.rand(1, 3, 8, 8),
        prompt_embeds=torch.rand(1, 4, 16),
        negative_prompt_embeds=torch.rand(1, 4, 16),
        latent=torch.rand(1, 16, 3, 1, 1),
        video=torch.rand(1, 3, 5, 8, 8),
    )


def test_wan_latent_clone_is_independent():
    original = _latent()
    clone = original.clone()
    clone.last_frame.add_(1.0)
    assert not torch.equal(original.last_frame, clone.last_frame)


def test_wan_latent_to_cpu_roundtrip():
    moved = _latent().to("cpu")
    assert moved.last_frame.device.type == "cpu"
    assert moved.latent is not None and moved.video is not None


def test_wan_latent_nelement_counts_all_tensors():
    latent = _latent()
    expected = sum(
        t.nelement()
        for t in (
            latent.last_frame,
            latent.prompt_embeds,
            latent.negative_prompt_embeds,
            latent.latent,
            latent.video,
        )
    )
    assert latent.nelement == expected


def test_wan_latent_handles_optional_none_fields():
    latent = WanLatent(last_frame=torch.rand(1, 3, 8, 8), prompt_embeds=torch.rand(1, 4, 16))
    assert latent.clone().video is None
    assert latent.to("cpu").latent is None


def test_pipeline_constructor_defaults():
    pipeline = WanI2VPipeline(repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert pipeline.is_loaded is False
    assert pipeline.pipeline_class == "WanImageToVideoPipeline"
    assert pipeline.flow_shift == 5.0


def test_pipeline_generate_before_load_raises():
    pipeline = WanI2VPipeline(repo="x")
    latent = WanLatent(last_frame=torch.rand(1, 3, 8, 8), prompt_embeds=torch.rand(1, 4, 16))
    with pytest.raises(AssertionError, match="not loaded"):
        pipeline.generate(latent, num_steps=4, guidance=5.0, num_frames=5, seed=0)


def test_pipeline_vram_estimate_is_positive():
    pipeline = WanI2VPipeline(repo="x")
    assert pipeline.profile_vram(height=480, width=832, num_frames=81) > 0
