r"""DreamDojo checkpoint download and DCP→.pt conversion."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

HF_REPO = "nvidia/DreamDojo"


def download_dreamdojo_checkpoint(ckpt_dir_name: str = "2B_pretrain") -> str:
    r"""Download DreamDojo checkpoint (DCP format) and convert to .pt."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{ckpt_dir_name}/**/model/*", f"{ckpt_dir_name}/latest_checkpoint.txt"],
        repo_type="model",
    )

    ckpt_dir = Path(local_dir) / ckpt_dir_name
    latest_file = ckpt_dir / "latest_checkpoint.txt"
    iter_name = latest_file.read_text().strip()
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
