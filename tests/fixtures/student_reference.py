r"""Provenance of the frozen student-streaming golden fixtures.

No distilled DreamDojo weights are public, but the student is architecturally
the teacher with causal per-frame execution (NVIDIA initializes the student
from teacher weights). This script produced
``tests/fixtures/data/student_mechanism_*`` by running the *reference*
streaming loop — ``generate_streaming_video`` / ``generate_next_frame`` /
``update_kv_cache`` from ``action_video2world_self_forcing.py``, bound to the
reference causal net loaded with real teacher weights. The native
`worldkernels.models.dreamdojo.pipeline.DreamDojoPipeline.generate_stream_latents`
reproduced it bit-exactly at capture time (2026-07-12).

Re-running requires a worldkernels checkout from before the
``cosmos_predict2`` wrapper deletion (git history:
``worldkernels/models/cosmos_predict2/deps.py``) plus the vendor repo clone.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data" / "student_mechanism_2b_240x320_seed7"
CKPT = "/home/aiengineer/.cache/worldkernels/dreamdojo/2B_pretrain/iter_000140000/model_ema_bf16.pt"
CACHE_FRAME_SIZE = 3
N_STEPS = 4
SEED = 7
H, W = 240, 320


def capture() -> Path:
    import torch
    import torch.nn as nn

    from worldkernels.models.cosmos_predict2 import deps as _deps

    _deps.ensure_cosmos_predict2()
    _deps._inject_stub(
        "transformer_engine.pytorch.distributed",
        {"get_all_rng_states": lambda: {}, "graph_safe_rng_available": lambda: False},
    )
    _deps._inject_stub("transformer_engine.pytorch.module", {})
    _deps._inject_stub(
        "transformer_engine.pytorch.module.base",
        {"TransformerEngineBaseModule": type("TransformerEngineBaseModule", (nn.Module,), {})},
    )

    import math
    from types import SimpleNamespace

    from cosmos_predict2._src.predict2.action.configs.action_conditioned.conditioner import (
        ActionConditionedCondition,
    )
    from cosmos_predict2._src.predict2.interactive.models.action_video2world_self_forcing import (
        ActionVideo2WorldModelTrigflowSelfForcingDMD2,
    )
    from cosmos_predict2._src.predict2.interactive.networks.dit_action_causal import (
        ActionChunkCausalDITwithConditionalMask,
    )
    from cosmos_predict2._src.predict2.modules.denoiser_scaling import RectifiedFlow_sCMWrapper
    from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

    from worldkernels.models.dreamdojo.checkpoint import remap_state_dict
    from worldkernels.models.dreamdojo.net import NET_2B

    net_kwargs = {k: v for k, v in NET_2B.items()}
    net = ActionChunkCausalDITwithConditionalMask(
        atten_backend="minimal_a2a",
        sac_config=SACConfig(mode="none"),
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        extra_per_block_abs_pos_emb=False,
        rope_enable_fps_modulation=False,
        action_dim=384,
        temporal_compression_ratio=4,
        timestep_scale=0.001,
        **net_kwargs,
    )
    sd = remap_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    net.load_state_dict(sd, strict=False)
    net = net.cuda().bfloat16().eval()

    model = ActionVideo2WorldModelTrigflowSelfForcingDMD2.__new__(
        ActionVideo2WorldModelTrigflowSelfForcingDMD2
    )
    nn.Module.__init__(model)
    model.net = net
    model.sigma_data = 1.0
    model.scaling_from_time = RectifiedFlow_sCMWrapper(1.0)
    model.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    model.config = SimpleNamespace(
        cache_frame_size=CACHE_FRAME_SIZE,
        selected_sampling_time=[
            math.pi / 2,
            math.atan(15.0),
            math.atan(5.0),
            math.atan(5.0 / 3.0),
        ],
        sigma_conditional=0.0001,
    )

    g = torch.Generator().manual_seed(SEED)
    h, w = H // 8, W // 8
    T = 4
    gt_latent = torch.randn(1, 16, T, h, w, generator=g).cuda().bfloat16()
    init_noise = torch.randn(1, 16, T, h, w, generator=g).cuda().float()
    text_emb = (torch.randn(1, 512, 100352, generator=g) * 0.05).cuda().bfloat16()
    action = (torch.rand(1, 12, 384, generator=g) * 2 - 1).cuda().bfloat16()
    padding_mask = torch.zeros(1, 1, H, W).cuda().bfloat16()
    zero_mask = torch.zeros(1, 1, 1, h, w).cuda().bfloat16()

    condition = ActionConditionedCondition(
        crossattn_emb=text_emb,
        action=action,
        gt_frames=gt_latent,
        condition_video_input_mask_B_C_T_H_W=zero_mask,
        padding_mask=padding_mask,
        fps=torch.tensor([4.0]).cuda().bfloat16(),
    )

    with torch.no_grad():
        latents = model.generate_streaming_video(
            condition,
            init_noise,
            n_steps=N_STEPS,
            cache_frame_size=CACHE_FRAME_SIZE,
            use_cuda_graphs=False,
            start_idx=1,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    tensors = {
        "gt_latent": gt_latent,
        "init_noise": init_noise,
        "text_emb": text_emb,
        "action": action,
        "output_latents": latents,
    }
    for name, t in tensors.items():
        save_file(
            {"tensor": t.detach().cpu().contiguous()}, str(OUTPUT_DIR / f"{name}.safetensors")
        )
    manifest = {
        "captured_at_unix": int(time.time()),
        "ckpt": CKPT,
        "cache_frame_size": CACHE_FRAME_SIZE,
        "n_steps": N_STEPS,
        "seed": SEED,
        "height": H,
        "width": W,
        "start_idx": 1,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"captured student mechanism fixtures to {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    sys.exit(0 if capture() else 1)
