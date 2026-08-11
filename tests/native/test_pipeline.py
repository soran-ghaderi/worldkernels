r"""Replay captured golden tensors through the native DreamDojo stack (GPU)."""

from __future__ import annotations

import pytest

from tests.native._assert import assert_close_to_reference, load_reference

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def loaded_pipeline(reference_dir, fixture_cfg):
    import torch

    from worldkernels.models.dreamdojo.pipeline import DreamDojoPipeline

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[fixture_cfg.dtype_str]
    return DreamDojoPipeline(
        fixture_cfg.variant, num_steps=fixture_cfg.num_steps, guidance=fixture_cfg.guidance
    ).load("cuda", dtype)


def test_generate_clip_matches_reference(reference_dir, fixture_cfg, loaded_pipeline):
    from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

    cond = load_reference(reference_dir, "cond_frame_uint8").cuda()
    action = load_reference(reference_dir, "pipeline_action").cuda()

    with set_forward_context(ForwardContext(attention_backend="sdpa")):
        latent, video = loaded_pipeline.generate_clip(
            cond,
            action,
            num_steps=fixture_cfg.num_steps,
            guidance=fixture_cfg.guidance,
            seed=fixture_cfg.seed,
        )

    ref_latent = load_reference(reference_dir, "pipeline_final_latent").cuda()
    assert_close_to_reference(latent, ref_latent, stage="pipeline_latent", name="final")

    uint8 = loaded_pipeline.to_uint8(video)
    ref_uint8 = load_reference(reference_dir, "pipeline_decoded_uint8").cuda()
    assert_close_to_reference(
        uint8.float(), ref_uint8.float(), stage="pipeline_decoded_uint8", name="video"
    )


def test_vae_roundtrip_matches_reference(reference_dir, loaded_pipeline):
    vae_in = load_reference(reference_dir, "vae_encode_input").cuda()
    import torch

    with torch.no_grad():
        enc = loaded_pipeline.tokenizer.encode(vae_in)
        dec = loaded_pipeline.tokenizer.decode(enc)
    assert_close_to_reference(
        enc.float(),
        load_reference(reference_dir, "vae_encode_output").cuda().float(),
        stage="vae_encode",
        name="encode",
    )
    assert_close_to_reference(
        dec.float(),
        load_reference(reference_dir, "vae_decode_output").cuda().float(),
        stage="vae_decode",
        name="decode",
    )
