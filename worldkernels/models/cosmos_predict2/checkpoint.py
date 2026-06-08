r"""Cosmos-Predict2.5-2B checkpoint resolution (scoped HF download).

The HF repo is ~75 GB: it bundles ~15 checkpoints (base/robot/auto) and stores one
pre-trained checkpoint in four redundant formats (DCP shards + ``model.pt`` +
``model_ema_fp32.pt`` + ``model_ema_bf16.pt``). A single usable 2B model is one
``*_ema_bf16.pt`` of ~4.1 GB. This module fetches only the selected checkpoint plus the
508 MB VAE tokenizer, so ``provision_weights`` never mirrors the whole repo.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

COSMOS_HF_REPO = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_VARIANT = "pretrained"

WAN_VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_VAE_FILE = "Wan2.1_VAE.pth"

CR1_EMPTY_TEXT_EMB_FILE = "robot/action-cond/cr1_empty_string_text_embeddings.pt"

COSMOS_CKPT_FILES = {
    "pretrained": "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
    "distilled": "base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt",
    "post-trained": "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
    # Deferred (action-conditioned / multiview — need the DreamDojo-style config, not the
    # plain video2world generator): robot/action-cond, robot/multiview-agibot, auto/multiview.
}


def resolve_ckpt_file(variant: str | None) -> str:
    return COSMOS_CKPT_FILES.get(variant or DEFAULT_VARIANT, COSMOS_CKPT_FILES[DEFAULT_VARIANT])


def download_vae_tokenizer() -> str:
    r"""Local path to the Wan2.1 VAE (the cosmos / DreamDojo tokenizer).

    cosmos's tokenizer *is* the Wan2.1 VAE. The ``nvidia/Cosmos-Predict2.5-2B`` mirror
    (``tokenizer.pth``) is a gated repo, so the identical ``Wan2.1_VAE.pth`` is fetched
    from the public Wan-AI repo, which works unauthenticated.
    """
    from huggingface_hub import hf_hub_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()
    return hf_hub_download(WAN_VAE_REPO, filename=WAN_VAE_FILE, repo_type="model")


def download_empty_text_embedding() -> str:
    r"""Local path to the precomputed CR1 empty-string text embedding.

    The action-conditioned model was trained and is evaluated with the real (non-zero)
    embedding of the empty prompt, not a zero tensor. With the 7B text encoder disabled
    this precomputed ``.pt`` supplies the in-distribution neutral conditioning instead.
    """
    from huggingface_hub import hf_hub_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()
    return hf_hub_download(COSMOS_HF_REPO, filename=CR1_EMPTY_TEXT_EMB_FILE, repo_type="model")


def provision_cosmos_weights(variant: str | None = None) -> str:
    r"""ModelCard ``weights_provider`` entry: fetch one ~4.1 GB checkpoint + the VAE tokenizer.

    Returns the local path to the selected ``_ema_bf16.pt`` and pre-caches ``tokenizer.pth`` so
    the pipeline's own ``hf_hub_download`` calls hit the cache. ~4.6 GB total vs 75 GB.
    """
    from huggingface_hub import hf_hub_download

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()

    ckpt_file = resolve_ckpt_file(variant)
    ckpt = hf_hub_download(COSMOS_HF_REPO, filename=ckpt_file, repo_type="model")
    download_vae_tokenizer()
    log.info("Cosmos-Predict2.5-2B: fetched %s + Wan2.1 VAE tokenizer (scoped)", ckpt_file)
    return ckpt
