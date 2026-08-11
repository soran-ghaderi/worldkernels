r"""CPU tests for the latent action encoder."""

from __future__ import annotations

import torch

from worldkernels.models.dreamdojo.lam import (
    LATENT_ACTION_DIM,
    LATENT_ACTION_SLICE,
    LatentActionEncoder,
)


def _tiny_encoder() -> LatentActionEncoder:
    return LatentActionEncoder(model_dim=32, patch_size=16, enc_blocks=2, num_heads=4)


def test_encode_pairs_shape():
    enc = _tiny_encoder().eval()
    pairs = torch.rand(3, 2, 48, 64, 3)
    with torch.no_grad():
        z = enc.encode_pairs(pairs)
    assert z.shape == (3, LATENT_ACTION_DIM)


def test_encode_video_resizes_and_pairs():
    enc = _tiny_encoder().eval()
    frames = (torch.rand(4, 3, 60, 80) * 255).to(torch.uint8)
    with torch.no_grad():
        z = enc.encode_video(frames)
    assert z.shape == (3, LATENT_ACTION_DIM)


def test_latent_action_slice():
    assert LATENT_ACTION_SLICE == slice(352, 384)


def test_state_dict_matches_reference_namespace():
    keys = set(_tiny_encoder().state_dict())
    expected = {
        "action_prompt",
        "fc.weight",
        "fc.bias",
        "encoder.ffn.0.weight",
        "encoder.ffn.1.weight",
        "encoder.ffn.2.weight",
        "encoder.out.weight",
        "encoder.transformer_blocks.0.spatial_attn.to_q.weight",
        "encoder.transformer_blocks.0.temporal_attn.rotary_embedding.freqs",
        "encoder.transformer_blocks.0.temporal_attn.to_out.0.weight",
        "encoder.transformer_blocks.1.ffn.3.bias",
        "encoder.transformer_blocks.1.norm3.weight",
    }
    missing = expected - keys
    assert not missing, f"missing: {sorted(missing)}"
    assert "encoder.pe" not in keys
