r"""DreamDojo checkpoint download, DCP→.pt conversion, and native net loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from worldkernels.models.dreamdojo.net import ActionVideoDiT

log = logging.getLogger(__name__)

HF_REPO = "nvidia/DreamDojo"
TEXT_EMB_REPO = "nvidia/Cosmos-Predict2.5-2B"
TEXT_EMB_FILE = "robot/action-cond/cr1_empty_string_text_embeddings.pt"

CKPT_DIRS = {
    "2b_pretrain": "2B_pretrain",
    "2b_gr1": "2B_GR1_post-train",
    "2b_agibot": "2B_AgiBot_post-train",
    "2b_g1": "2B_G1_post-train",
    "2b_yam": "2B_YAM_post-train",
    "14b_pretrain": "14B_pretrain",
    "14b_gr1": "14B_GR1_post-train",
    "14b_agibot": "14B_AgiBot_post-train",
    "14b_g1": "14B_G1_post-train",
    "14b_yam": "14B_YAM_post-train",
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


def download_lam_checkpoint() -> str:
    r"""Local path to the latent action model checkpoint (``LAM_400k.ckpt``)."""
    from huggingface_hub import hf_hub_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()
    return hf_hub_download(HF_REPO, filename="LAM_400k.ckpt", repo_type="model")


def download_text_embedding() -> str:
    r"""Local path to the precomputed CR1 empty-string text embedding.

    The action-conditioned model was trained and is evaluated with the real
    (non-zero) embedding of the empty prompt; this precomputed ``.pt`` supplies
    that in-distribution neutral conditioning without the 7B text encoder. A
    locally computed copy in the worldkernels cache (see
    ``compute_text_embedding``) takes precedence over the gated HF download.
    """
    from worldkernels.bootstrap import cache

    local = cache.home() / "dreamdojo" / "cr1_empty_string_text_embeddings.pt"
    if local.exists():
        return str(local)

    from huggingface_hub import hf_hub_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()
    try:
        return hf_hub_download(TEXT_EMB_REPO, filename=TEXT_EMB_FILE, repo_type="model")
    except Exception as err:
        raise RuntimeError(
            f"Cannot fetch the CR1 empty-string text embedding from the gated repo "
            f"{TEXT_EMB_REPO!r} (accept its license and set a valid HF_TOKEN), or run "
            f"compute_text_embedding() once to derive it locally from "
            f"nvidia/Cosmos-Reason1-7B."
        ) from err


def compute_text_embedding() -> str:
    r"""Derive the CR1 empty-string embedding locally; returns the cached ``.pt`` path.

    Reproduces the reference recipe: the Cosmos-Reason1-7B chat template around
    the empty prompt, padded to 512 tokens, hidden states of all 28 layers each
    mean-normalized per token and concatenated to 100352 channels.
    """
    prompt = ""
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    from worldkernels.bootstrap import cache

    repo = "nvidia/Cosmos-Reason1-7B"
    tok = AutoTokenizer.from_pretrained(repo)
    system_text = "You are a helpful assistant who will provide prompts to an image generator."
    conversations: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = tok.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False)["input_ids"][:512]
    ids = ids + [tok.pad_token_id] * (512 - len(ids))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)

    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # type: ignore[arg-type]
        repo, torch_dtype=torch.bfloat16
    )
    model = encoder.to(device).eval()  # type: ignore[arg-type]
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    def _mean_normalize(t: "torch.Tensor") -> "torch.Tensor":
        return (t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-8)

    emb = torch.cat([_mean_normalize(h) for h in hidden_states[1:]], dim=-1)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    out_path = cache.home() / "dreamdojo" / "cr1_empty_string_text_embeddings.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb.squeeze(0).float().cpu(), out_path)
    return str(out_path)


def remap_state_dict(state_dict: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    r"""Converted-checkpoint keys → native `ActionVideoDiT` namespace.

    Strips the ``net.`` prefix and drops TransformerEngine ``*_extra_state``
    blobs and ``accum_*`` training counters. Applies unchanged to teacher and
    distilled-student checkpoints (shared namespace).
    """
    out = {}
    for key, value in state_dict.items():
        if key.endswith("_extra_state"):
            continue
        key = key.removeprefix("net.")
        if key.startswith("accum_"):
            continue
        out[key] = value
    return out


def load_net(
    variant: str,
    *,
    ckpt_path: str | None = None,
    device: str | "torch.device" = "cuda",
    dtype: "torch.dtype | None" = None,
) -> "ActionVideoDiT":
    r"""Build the native DreamDojo DiT for ``variant`` and load its weights.

    Args:
        variant: A `CKPT_DIRS` key (e.g. ``"2b_gr1"``); the size prefix picks
            the net preset.
        ckpt_path: Optional converted ``.pt`` path (e.g. a distilled student
            checkpoint); defaults to downloading the released teacher weights.
    """
    import torch

    from worldkernels.models.dreamdojo.net import NET_2B, NET_14B, ActionVideoDiT

    if variant not in CKPT_DIRS:
        raise ValueError(f"unknown DreamDojo variant {variant!r}; known: {sorted(CKPT_DIRS)}")
    preset = NET_14B if variant.startswith("14b") else NET_2B
    with torch.device("meta"):
        net = ActionVideoDiT(**preset)

    path = ckpt_path or download_dreamdojo_checkpoint(CKPT_DIRS[variant])
    state_dict = remap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.load_state_dict(state_dict, assign=True)
    net = net.to(device=device, dtype=dtype or torch.bfloat16)
    return net.eval().requires_grad_(False)
