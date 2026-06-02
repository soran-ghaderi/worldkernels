r"""DreamDojo checkpoint download and DCP→.pt conversion."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

HF_REPO = "nvidia/DreamDojo"


def download_dreamdojo_checkpoint(ckpt_dir_name: str = "2B_pretrain") -> str:
    r"""Download DreamDojo checkpoint (DCP format) and convert to .pt.

    Two-stage fetch so we only pull the latest iteration's shards, not every
    training checkpoint in the repo. Stage 1 grabs ``latest_checkpoint.txt``
    (a few bytes); stage 2 grabs only ``{ckpt_dir}/{latest_iter}/model/*``.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()

    latest_file_local = hf_hub_download(
        HF_REPO,
        filename=f"{ckpt_dir_name}/latest_checkpoint.txt",
        repo_type="model",
    )
    iter_name = Path(latest_file_local).read_text().strip()
    log.info(
        "DreamDojo %s: latest iteration is %r — fetching only that iter's shards",
        ckpt_dir_name,
        iter_name,
    )

    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{ckpt_dir_name}/{iter_name}/model/*"],
        repo_type="model",
    )

    ckpt_dir = Path(local_dir) / ckpt_dir_name
    iter_dir = ckpt_dir / iter_name

    pt_path = iter_dir / "model_ema_bf16.pt"
    if pt_path.exists():
        return str(pt_path)

    log.info("Converting DCP checkpoint to .pt at %s", pt_path)
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    full_pt = iter_dir / "model.pt"
    dcp_to_torch_save(iter_dir / "model", full_pt)

    state_dict = torch.load(full_pt, map_location="cpu", weights_only=False)
    ema_bf16 = {}
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            new_key = key.replace("net_ema.", "net.")
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                value = value.bfloat16()
            ema_bf16[new_key] = value
    torch.save(ema_bf16, pt_path)
    full_pt.unlink()
    log.info("Saved EMA bf16 checkpoint: %s", pt_path)
    return str(pt_path)
