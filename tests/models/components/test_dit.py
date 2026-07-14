r"""CPU tests for DiT blocks and the assembled VideoDiT backbone."""

from __future__ import annotations

import pytest
import torch

from worldkernels.models.components.dit import DiTBlock, FeedForward, FinalLayer, VideoDiT


def _tiny_dit(**overrides) -> VideoDiT:
    kwargs = dict(
        max_img_h=32,
        max_img_w=32,
        max_frames=8,
        in_channels=16,
        out_channels=16,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        adaln_lora_dim=16,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=48,
        crossattn_emb_channels=32,
        rope_h_extrapolation_ratio=3.0,
        rope_w_extrapolation_ratio=3.0,
    )
    kwargs.update(overrides)
    return VideoDiT(**kwargs)


def test_feedforward_no_bias():
    ff = FeedForward(8, 32)
    assert ff.layer1.bias is None and ff.layer2.bias is None
    assert ff(torch.randn(2, 3, 8)).shape == (2, 3, 8)


def test_block_forward_shape():
    block = DiTBlock(64, 32, num_heads=4, use_adaln_lora=True, adaln_lora_dim=16)
    x = torch.randn(1, 2, 4, 4, 64)
    emb = torch.randn(1, 2, 64)
    lora = torch.randn(1, 2, 192)
    ctx = torch.randn(1, 5, 32)
    rope = torch.zeros(2 * 4 * 4, 1, 1, 16)
    out = block(x, emb, ctx, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=lora)
    assert out.shape == x.shape


def test_final_layer_shape():
    fl = FinalLayer(64, 2, 1, 16, use_adaln_lora=True, adaln_lora_dim=16)
    x = torch.randn(1, 2, 4, 4, 64)
    emb = torch.randn(1, 2, 64)
    lora = torch.randn(1, 2, 192)
    assert fl(x, emb, adaln_lora_B_T_3D=lora).shape == (1, 2, 4, 4, 2 * 2 * 16)


def test_videodit_forward_roundtrip_shape():
    net = _tiny_dit()
    x = torch.randn(1, 16, 2, 16, 16)
    t = torch.rand(1, 2)
    ctx = torch.randn(1, 5, 48)
    pm = torch.zeros(1, 1, 16, 16)
    out = net(x, t, ctx, padding_mask=pm)
    assert out.shape == (1, 16, 2, 16, 16)


def test_videodit_state_dict_matches_reference_namespace():
    r"""Key layout must stay loadable from remapped Cosmos-Predict2.5 checkpoints."""
    keys = set(_tiny_dit().state_dict())
    expected_subset = {
        "x_embedder.proj.1.weight",
        "pos_embedder.seq",
        "pos_embedder.dim_spatial_range",
        "pos_embedder.dim_temporal_range",
        "t_embedder.1.linear_1.weight",
        "t_embedder.1.linear_2.weight",
        "t_embedding_norm.weight",
        "crossattn_proj.0.weight",
        "crossattn_proj.0.bias",
        "final_layer.linear.weight",
        "final_layer.adaln_modulation.1.weight",
        "final_layer.adaln_modulation.2.weight",
        "blocks.0.self_attn.q_proj.weight",
        "blocks.0.self_attn.q_norm.weight",
        "blocks.0.self_attn.output_proj.weight",
        "blocks.0.cross_attn.k_proj.weight",
        "blocks.0.mlp.layer1.weight",
        "blocks.0.mlp.layer2.weight",
        "blocks.0.adaln_modulation_self_attn.1.weight",
        "blocks.0.adaln_modulation_cross_attn.2.weight",
        "blocks.0.adaln_modulation_mlp.1.weight",
        "blocks.1.self_attn.k_norm.weight",
    }
    missing = expected_subset - keys
    assert not missing, f"missing reference keys: {sorted(missing)}"


def test_videodit_requires_padding_mask():
    net = _tiny_dit()
    x = torch.randn(1, 16, 1, 16, 16)
    with pytest.raises(AssertionError):
        net(x, torch.rand(1, 1), torch.randn(1, 5, 48))


def test_patch_size_divisibility_enforced():
    net = _tiny_dit()
    x = torch.randn(1, 16, 1, 15, 16)
    with pytest.raises(AssertionError):
        net(x, torch.rand(1, 1), torch.randn(1, 5, 48), padding_mask=torch.zeros(1, 1, 15, 16))
