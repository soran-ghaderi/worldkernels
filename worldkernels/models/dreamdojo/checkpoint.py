r"""DreamDojo checkpoint download and DCP→.pt conversion."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

HF_REPO = "nvidia/DreamDojo"

CKPT_DIRS = {
    "2b_pretrain": "2B_pretrain",
    "2b_gr1": "2B_GR1_post-train",
    "2b_agibot": "2B_AgiBot_post-train",
    "2b_g1": "2B_G1_post-train",
    "2b_yam": "2B_YAM_post-train",
    "14b_pretrain": "14B_pretrain",
    "14b_gr1": "14B_GR1_post-train",
}


def provision_dreamdojo_weights(variant: str | None = None) -> str:
    r"""ModelCard ``weights_provider`` entry: map a variant to its checkpoint dir.

    The DreamDojo repo is an ~859 GB monorepo (every 2B/14B variant, each with
    model + optimizer + scheduler + trainer shards). A blanket ``snapshot_download``
    would mirror all of it, so the card routes provisioning here, which fetches only
    the requested variant's latest model shards (~13 GB for 2B).
    """
    return download_dreamdojo_checkpoint(CKPT_DIRS.get(variant or "2b_pretrain", "2B_pretrain"))


def download_dreamdojo_checkpoint(ckpt_dir_name: str = "2B_pretrain") -> str:
    r"""Download a DreamDojo checkpoint (DCP format) and convert to a bf16 ``.pt``.

    DreamDojo ships only DCP training shards (no ready-made inference file). The latest
    iter's ``model/`` directory is ~12.9 GB: ``net_ema.*`` (the EMA, fp32, ~8.6 GB) +
    ``net.*`` (the model, ~4.3 GB), FSDP-sharded uniformly across all 8 ``.distcp`` files
    so none can be skipped. We fetch those shards into a scratch dir, keep only the EMA
    weights converted to bf16 (~4.3 GB), then **delete the scratch shards** so the cached
    footprint is the converted file alone, not ~17 GB.

    Two-stage fetch: stage 1 grabs ``latest_checkpoint.txt`` (a few bytes); stage 2 grabs
    only ``{ckpt_dir}/{latest_iter}/model/*`` (the sibling ``optim/`` Adam-moment shards
    are deliberately excluded).
    """
    import shutil

    from huggingface_hub import hf_hub_download, snapshot_download

    from worldkernels.bootstrap import cache
    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()

    latest_file_local = hf_hub_download(
        HF_REPO,
        filename=f"{ckpt_dir_name}/latest_checkpoint.txt",
        repo_type="model",
    )
    iter_name = Path(latest_file_local).read_text().strip()
    if not iter_name or "/" in iter_name or iter_name.startswith("."):
        raise RuntimeError(
            f"DreamDojo {ckpt_dir_name}: invalid latest iter {iter_name!r} in latest_checkpoint.txt"
        )

    out_dir = cache.home() / "dreamdojo" / ckpt_dir_name / iter_name
    pt_path = out_dir / "model_ema_bf16.pt"
    if pt_path.exists():
        return str(pt_path)

    log.info(
        "DreamDojo %s: latest iteration is %r — fetching only that iter's model shards",
        ckpt_dir_name,
        iter_name,
    )
    work_dir = out_dir / "_dcp"
    snapshot_download(
        HF_REPO,
        allow_patterns=[f"{ckpt_dir_name}/{iter_name}/model/*"],
        repo_type="model",
        local_dir=str(work_dir),
    )
    model_dir = work_dir / ckpt_dir_name / iter_name / "model"
    if not model_dir.is_dir() or not any(model_dir.glob("*.distcp")):
        raise RuntimeError(
            f"DreamDojo {ckpt_dir_name}/{iter_name}: no model shards under {model_dir} "
            "(repo layout changed?)"
        )

    log.info("Converting DCP checkpoint to bf16 EMA .pt at %s", pt_path)
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = _consolidate_dcp(model_dir, work_dir)
    ema_bf16 = {}
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            new_key = "net." + key[len("net_ema.") :]
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                value = value.bfloat16()
            ema_bf16[new_key] = value
    if not ema_bf16:
        raise RuntimeError(
            f"DreamDojo {ckpt_dir_name}/{iter_name}: checkpoint has no 'net_ema.*' weights"
        )
    torch.save(ema_bf16, pt_path)
    shutil.rmtree(work_dir, ignore_errors=True)
    log.info(
        "Saved EMA bf16 checkpoint (%.1f GB) and pruned raw DCP shards: %s",
        pt_path.stat().st_size / 1e9,
        pt_path,
    )
    return str(pt_path)


def _consolidate_dcp(model_dir: Path, scratch_dir: Path) -> dict:
    r"""DCP shards → a consolidated CPU state dict (via a transient ``.pt`` in ``scratch_dir``)."""
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    full_pt = scratch_dir / "model.pt"
    try:
        dcp_to_torch_save(str(model_dir), str(full_pt))
        return torch.load(full_pt, map_location="cpu", weights_only=False)
    finally:
        full_pt.unlink(missing_ok=True)
