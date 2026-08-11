r"""CPU tests for the native DreamDojo action-conditioned DiT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from worldkernels.models.dreamdojo.net import NET_2B, NET_14B, ActionEmbedder, ActionVideoDiT

DATA = Path(__file__).parent / "data"

TINY: dict[str, Any] = dict(
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
    action_dim=6,
    hidden_dim_in_action_embedder=32,
)


@pytest.mark.parametrize(
    ("preset", "manifest"),
    [(NET_2B, "keys_2b.json"), (NET_14B, "keys_14b.json")],
    ids=["2b", "14b"],
)
def test_state_dict_matches_released_checkpoint_manifest(preset, manifest):
    r"""Native key/shape layout must exactly cover the released DCP checkpoints."""
    reference = json.loads((DATA / manifest).read_text())["keys"]
    with torch.device("meta"):
        net = ActionVideoDiT(**preset)
    sd = net.state_dict()
    assert set(sd) == set(reference)
    mismatched = {k for k in sd if list(sd[k].shape) != reference[k]}
    assert not mismatched, f"shape mismatches: {sorted(mismatched)[:8]}"


def test_action_embedder_layout():
    emb = ActionEmbedder(24, 32, 8)
    assert emb.fc1.bias is not None and emb.fc2.bias is not None
    assert emb(torch.randn(2, 3, 24)).shape == (2, 3, 8)


def test_action_chunks_zero_padded_for_conditioning_frame():
    net = ActionVideoDiT(**TINY)
    action = torch.randn(2, 8, 6)
    emb_d, emb_3d = net._embed_action_chunks(action)
    assert emb_d.shape == (2, 3, 64)
    assert emb_3d.shape == (2, 3, 192)
    assert torch.all(emb_d[:, 0] == 0) and torch.all(emb_3d[:, 0] == 0)

    flat0 = action[:, :4].reshape(2, 1, 24)
    torch.testing.assert_close(emb_d[:, 1:2], net.action_embedder_B_D(flat0))


def test_teacher_forward_shape():
    net = ActionVideoDiT(**TINY)
    x = torch.randn(1, 16, 3, 16, 16)
    t = torch.rand(1, 3) * 1000
    ctx = torch.randn(1, 5, 48)
    mask = torch.zeros(1, 1, 3, 16, 16)
    mask[:, :, :1] = 1.0
    action = torch.randn(1, 8, 6)
    pm = torch.zeros(1, 1, 16, 16)
    out = net(x, t, ctx, condition_video_input_mask_B_C_T_H_W=mask, action=action, padding_mask=pm)
    assert out.shape == (1, 16, 3, 16, 16)


def test_forward_frame_shape_and_cache_write():
    torch.manual_seed(0)
    net = ActionVideoDiT(**TINY).eval()
    x = torch.randn(1, 16, 1, 16, 16)
    t = torch.rand(1, 1) * 1000
    ctx = torch.randn(1, 5, 48)
    mask = torch.ones(1, 1, 1, 16, 16)
    pm = torch.zeros(1, 1, 16, 16)

    caches = net.allocate_kv_caches(
        max_frames=4,
        batch_size=1,
        latent_h=16,
        latent_w=16,
        window=3,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(caches) == 2
    with torch.no_grad():
        out = net.forward_frame(
            x,
            0,
            t,
            ctx,
            condition_video_input_mask_B_C_1_H_W=mask,
            action=None,
            kv_caches=caches,
            store_kv=True,
            padding_mask=pm,
        )
    assert out.shape == (1, 16, 1, 16, 16)
    assert not torch.all(caches[0].k[0] == 0)
    assert torch.all(caches[0].k[1] == 0)


def test_forward_frame_uses_cached_context():
    torch.manual_seed(1)
    net = ActionVideoDiT(**TINY).eval()
    ctx = torch.randn(1, 5, 48)
    pm = torch.zeros(1, 1, 16, 16)
    mask = torch.zeros(1, 1, 1, 16, 16)
    action = torch.randn(1, 4, 6)

    def run(with_history: bool) -> torch.Tensor:
        caches = net.allocate_kv_caches(
            max_frames=4,
            batch_size=1,
            latent_h=16,
            latent_w=16,
            window=3,
            device="cpu",
            dtype=torch.float32,
        )
        torch.manual_seed(2)
        if with_history:
            frame0 = torch.randn(1, 16, 1, 16, 16)
            with torch.no_grad():
                net.forward_frame(
                    frame0,
                    0,
                    torch.zeros(1, 1),
                    ctx,
                    condition_video_input_mask_B_C_1_H_W=mask,
                    kv_caches=caches,
                    store_kv=True,
                    padding_mask=pm,
                )
        x1 = torch.ones(1, 16, 1, 16, 16)
        with torch.no_grad():
            return net.forward_frame(
                x1,
                1 if with_history else 0,
                torch.full((1, 1), 500.0),
                ctx,
                condition_video_input_mask_B_C_1_H_W=mask,
                action=action,
                kv_caches=caches,
                store_kv=False,
                padding_mask=pm,
            )

    assert not torch.allclose(run(True), run(False))
