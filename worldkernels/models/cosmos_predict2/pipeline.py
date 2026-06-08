r"""Video generator backed by NVIDIA's cosmos_predict2 package.

Wraps cosmos_predict2 for inference: model loading + path patching, text/image
encoding, the denoising loop, and VAE decode. Implements the
`VideoGenerator` contract so it can be wrapped
by `GeneratorWorld`; it is also composed
directly by the action-conditioned `DreamDojoWorld`.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.models.base import GenerationResult, VideoGenerator
from worldkernels.models.cosmos_predict2.checkpoint import (
    COSMOS_HF_REPO,
    DEFAULT_VARIANT,
    download_empty_text_embedding,
    download_vae_tokenizer,
    resolve_ckpt_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

DEFAULT_COSMOS_CONFIG = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
DEFAULT_COSMOS_EXPERIMENT = (
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16"
    "-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2"
)
FALLBACK_CONFIG = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
FALLBACK_EXPERIMENT = "dreamdojo_2b_480_640_pretrain"


def _download_hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


class CosmosPredict2Latent:
    r"""Conditioning state for cosmos_predict2-backed generation."""

    __slots__ = ("latent", "last_frame", "text_emb", "neg_text_emb")

    def __init__(
        self,
        latent: torch.Tensor,
        last_frame: torch.Tensor,
        text_emb: torch.Tensor,
        neg_text_emb: torch.Tensor | None = None,
    ) -> None:
        self.latent = latent
        self.last_frame = last_frame
        self.text_emb = text_emb
        self.neg_text_emb = neg_text_emb

    def clone(self) -> CosmosPredict2Latent:
        return CosmosPredict2Latent(
            latent=self.latent.clone(),
            last_frame=self.last_frame.clone(),
            text_emb=self.text_emb.clone(),
            neg_text_emb=self.neg_text_emb.clone() if self.neg_text_emb is not None else None,
        )

    def to(self, device: Any) -> CosmosPredict2Latent:
        return CosmosPredict2Latent(
            latent=self.latent.to(device),
            last_frame=self.last_frame.to(device),
            text_emb=self.text_emb.to(device),
            neg_text_emb=self.neg_text_emb.to(device) if self.neg_text_emb is not None else None,
        )

    @property
    def nelement(self) -> int:
        n = self.latent.nelement() + self.last_frame.nelement() + self.text_emb.nelement()
        if self.neg_text_emb is not None:
            n += self.neg_text_emb.nelement()
        return n

    @property
    def element_size(self) -> int:
        return self.latent.element_size()


class CosmosPredict2Pipeline(VideoGenerator):
    r"""Video generator driving cosmos_predict2 for inference.

    Args:
        experiment: cosmos_predict2 experiment config name.
        config_file: cosmos_predict2 config module path.
        ckpt_path: Optional explicit checkpoint; resolved by HF download on
            `load()` when omitted.
    """

    LATENT_CH: int = 16
    SPATIAL_FACTOR: int = 8
    NEGATIVE_PROMPT: str = (
        "The video captures a series of frames showing ugly scenes, static with no motion, "
        "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
        "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
        "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
        "artifacting, color banding, unnatural transitions, outdated special effects, fake "
        "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, "
        "and flickering. Overall, the video is of poor quality."
    )

    def __init__(
        self,
        *,
        experiment: str = DEFAULT_COSMOS_EXPERIMENT,
        config_file: str = DEFAULT_COSMOS_CONFIG,
        ckpt_path: str | None = None,
        variant: str = DEFAULT_VARIANT,
        text_encoder: str = "auto",
    ) -> None:
        if text_encoder not in ("auto", "on", "off"):
            raise ValueError(f"text_encoder must be auto|on|off, got {text_encoder!r}")
        self.experiment = experiment
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.variant = variant
        self.text_encoder_mode = text_encoder
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._model: Any = None
        self._neg_text_emb: torch.Tensor | None = None
        self._loaded = False
        self._stub_text_emb_dim: int = 0
        self._empty_text_emb: torch.Tensor | None = None
        self._text_encoder: Any = None
        self._text_encoder_cfg: Any = None

    @property
    def _text_encoder_enabled(self) -> bool:
        r"""Whether the 16.6 GB reason1-7B text encoder should be loaded + used.

        Only ``text_encoder="on"`` loads it. ``auto`` (default) and ``off`` skip it so neither
        the encoder nor its weights are downloaded; ``encode_text`` returns synthetic embeddings
        and inference still runs (unconditional / action-conditioned).
        """
        return self.text_encoder_mode == "on" and not os.environ.get("WK_STUB_TEXT_ENCODER")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def neg_text_emb(self) -> torch.Tensor | None:
        return self._neg_text_emb

    def load(self, device: str, dtype: torch.dtype, ckpt_path: str | None = None) -> None:
        r"""Load model weights and prepare the pipeline for inference.

        ``ckpt_path`` may be passed explicitly (used by DreamDojo); otherwise
        the checkpoint is resolved from the configured path or downloaded.
        """
        from worldkernels.models.cosmos_predict2.deps import ensure_cosmos_predict2

        ensure_cosmos_predict2()

        self.device = device
        self.dtype = dtype

        ckpt = ckpt_path or self.ckpt_path
        if ckpt is not None:
            experiment, config_file = self.experiment, self.config_file
        else:
            ckpt, experiment, config_file = self._resolve_default_checkpoint()

        log.info("Loading model (experiment=%s)", experiment)
        self._model = self._load_model(experiment, ckpt, config_file)
        if self._text_encoder_enabled:
            self._neg_text_emb = self.encode_text(self.NEGATIVE_PROMPT)
        else:
            log.info(
                "Text encoder not loaded (text_encoder=%s); prompts use synthetic embeddings. "
                "Pass text_encoder='on' for prompt-conditioned generation.",
                self.text_encoder_mode,
            )
        self._loaded = True
        log.info(
            "CosmosPredict2 pipeline loaded on %s (%.1f GB VRAM)",
            device,
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        )

    def _resolve_default_checkpoint(self) -> tuple[str, str, str]:
        r"""Resolve checkpoint + experiment + config, falling back to DreamDojo pretrain."""
        try:
            log.info("Downloading Cosmos-Predict2.5-2B checkpoint (variant=%s)...", self.variant)
            ckpt = _download_hf_file(COSMOS_HF_REPO, resolve_ckpt_file(self.variant))
            return ckpt, self.experiment, self.config_file
        except Exception as exc:
            log.info(
                "Cosmos-Predict2.5-2B unavailable (%s); falling back to DreamDojo pretrain",
                exc.__class__.__name__,
            )
            from worldkernels.models.dreamdojo.checkpoint import download_dreamdojo_checkpoint

            return download_dreamdojo_checkpoint(), FALLBACK_EXPERIMENT, FALLBACK_CONFIG

    def _load_model(self, experiment: str, ckpt_path: str, config_file: str) -> Any:
        from cosmos_predict2._src.imaginaire.lazy_config import instantiate
        from cosmos_predict2._src.imaginaire.utils import misc
        from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
        from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

        config_module = get_config_module(config_file)
        config = importlib.import_module(config_module).make_config()
        config = override(
            config,
            [
                "--",
                f"experiment={experiment}",
                "data_train=mock",
                "data_val=mock",
            ],
        )

        config.checkpoint.load_path = str(ckpt_path)
        config.model.config.ema.enabled = False
        config.model.config.fsdp_shard_size = 1

        self._stub_text_emb_dim = config.model.config.net.crossattn_proj_in_channels
        self._text_encoder_cfg = None
        if self._text_encoder_enabled:
            import copy

            from omegaconf import OmegaConf

            self._patch_text_encoder_paths()
            tec = copy.deepcopy(config.model.config.text_encoder_config)
            qcfg = tec.model_config.model_config
            OmegaConf.set_struct(qcfg, False)
            qcfg.attn_implementation = "sdpa"
            qcfg.attn_implementation_autoset = False
            self._text_encoder_cfg = tec
        config.model.config.text_encoder_config = None

        self._patch_tokenizer_path()

        config.validate()
        config.freeze()
        misc.set_random_seed(seed=0, by_rank=True)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

        log.info("Instantiating model...")
        model = instantiate(config.model)
        model.on_train_start()

        log.info("Loading weights from %s", ckpt_path)
        state_dict = easy_io.load(ckpt_path, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        torch.cuda.empty_cache()
        return model

    def _patch_tokenizer_path(self) -> None:
        r"""Point the Wan2.1 VAE at a local checkpoint so it never hits cosmos's ``s3://`` path.

        Without this the VAE loads from an internal object store via the stubbed
        ``multistorageclient`` backend and fails. Raises if the (non-gated) tokenizer cannot
        be fetched, rather than silently letting the run fail later in the VAE.
        """
        tokenizer_path = download_vae_tokenizer()

        from cosmos_predict2._src.predict2.tokenizers import wan2pt1

        _orig_init = wan2pt1.WanVAE.__init__

        def _patched_init(self_vae, vae_pth=tokenizer_path, **kwargs):
            return _orig_init(self_vae, vae_pth=tokenizer_path, **kwargs)

        wan2pt1.WanVAE.__init__ = _patched_init

    @staticmethod
    def _patch_text_encoder_paths() -> None:
        from cosmos_predict2._src.imaginaire.utils import checkpoint_db

        _orig = checkpoint_db.get_checkpoint_path

        def _patched(uri: str) -> str:
            if uri.startswith("s3://") and ("sft_exp721" in uri or "cosmos_reasoning1" in uri):
                from huggingface_hub import snapshot_download

                return snapshot_download("nvidia/Cosmos-Reason1-7B", repo_type="model")
            return uri if uri.startswith("s3://") else _orig(uri)

        checkpoint_db.get_checkpoint_path = _patched

        from cosmos_predict2._src.reason1.tokenizer import processor as proc_mod

        _orig_proc = proc_mod.Processor.__init__

        def _patched_proc(self_proc, tokenizer_type="qwen2_5_vl", cache_dir=None):
            if cache_dir and str(cache_dir).startswith("s3://"):
                cache_dir = "Qwen/Qwen2.5-VL-7B-Instruct"
            return _orig_proc(self_proc, tokenizer_type, cache_dir)

        proc_mod.Processor.__init__ = _patched_proc

    def _stub_text_emb(self, prompt: str, *, seq_len: int = 16) -> torch.Tensor:
        r"""Zero text embedding — fallback only, used when the real empty-string embedding
        cannot be fetched. The model was trained with the non-zero empty-string embedding,
        so zeros are out-of-distribution and degrade action/image-conditioned rollouts.
        """
        return torch.zeros(
            1, seq_len, self._stub_text_emb_dim, device=self.device, dtype=self.dtype
        )

    def _empty_string_emb(self) -> torch.Tensor:
        r"""Real (non-zero) CR1 empty-string embedding, cached; zero fallback if unavailable.

        This is the in-distribution neutral conditioning the action-conditioned model expects
        (vendor inference loads the same precomputed tensor). Fetched once from the cosmos repo,
        moved to the pipeline device/dtype, and reshaped to ``[1, T, D]``.
        """
        if self._empty_text_emb is not None:
            return self._empty_text_emb
        try:
            emb = torch.load(download_empty_text_embedding(), map_location="cpu")
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            emb = emb.to(device=self.device, dtype=self.dtype)
            log.info(
                "Using real CR1 empty-string text embedding: shape=%s, nonzero=%d",
                tuple(emb.shape),
                int(torch.count_nonzero(emb).item()),
            )
            if self._stub_text_emb_dim and emb.shape[-1] != self._stub_text_emb_dim:
                log.warning(
                    "CR1 embedding dim %d != model crossattn_proj_in_channels %d; the forward "
                    "may error or produce garbage. This is likely the wrong embedding file.",
                    emb.shape[-1],
                    self._stub_text_emb_dim,
                )
        except Exception as exc:
            log.warning(
                "CR1 empty-string embedding unavailable (%r); falling back to a zero text "
                "embedding, which may degrade quality. Use text_encoder='on' for real prompts.",
                exc,
            )
            emb = self._stub_text_emb("")
        self._empty_text_emb = emb
        return emb

    def encode_text(self, prompt: str) -> torch.Tensor:
        r"""Compute the text embedding for ``prompt``.

        With the text encoder disabled (``text_encoder`` auto/off, or ``WK_STUB_TEXT_ENCODER``)
        this returns the real empty-string embedding (the 7B reason1 encoder is never loaded) so
        action/image-conditioned inference runs with in-distribution neutral conditioning. With
        ``text_encoder='on'`` the encoder is built once on CPU and reused for real prompts.
        ``WK_STUB_TEXT_ENCODER`` forces the offline zero stub (deterministic, no network) for
        capture/tests.
        """
        if not self._text_encoder_enabled:
            if os.environ.get("WK_STUB_TEXT_ENCODER"):
                return self._stub_text_emb(prompt)
            return self._empty_string_emb()
        emb = self._get_text_encoder().compute_text_embeddings_online(
            data_batch={"ai_caption": [prompt], "images": None},
            input_caption_key="ai_caption",
        )
        return emb.to(device=self.device, dtype=self.dtype)

    def _get_text_encoder(self) -> Any:
        r"""Lazily build the reason1 text encoder on CPU, offloaded from the GPU DiT.

        The cosmos encoder hardcodes ``flash_attention_2`` and moves inputs to ``cuda``; ``sdpa``
        is forced at config time (the flash_attn stub raises if called) and inputs are coerced to
        the encoder's device, so the 7B encoder runs on CPU and never competes with the DiT for
        the 24 GB of VRAM.
        """
        if self._text_encoder is not None:
            return self._text_encoder
        if self._text_encoder_cfg is None:
            raise RuntimeError(
                "text_encoder='on' but the encoder config was not captured at load()"
            )
        from worldkernels.models.cosmos_predict2.deps import install_reason1_transformers_compat

        install_reason1_transformers_compat()
        from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder

        log.info("Building reason1 text encoder on CPU (one-time; offloaded from the GPU DiT)...")
        try:
            enc = TextEncoder(self._text_encoder_cfg, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                "Could not build the reason1-7B text encoder. NVIDIA's reason1 encoder targets an "
                f"older transformers and is incompatible with the installed version ({exc!r}). Use "
                "text_encoder='off' (the default) for image + action conditioning, which needs no "
                "text encoder, or install a compatible transformers to enable real text."
            ) from exc
        _orig_forward = enc.model.forward

        def _forward_on_model_device(input_ids: Any, *args: Any, **kwargs: Any) -> Any:
            dev = next(enc.model.parameters()).device
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(dev)
            return _orig_forward(input_ids, *args, **kwargs)

        enc.model.forward = _forward_on_model_device
        self._text_encoder = enc
        return enc

    def encode_image(
        self, image: Any, *, height: int, width: int, frames_per_step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Encode an initial image into ``(last_frame, latent)``."""
        import torchvision.transforms.functional as TF

        if isinstance(image, str):
            from PIL import Image

            img = Image.open(image).convert("RGB")
            img_t = TF.to_tensor(img.resize((width, height)))
        else:
            img_t = torch.as_tensor(image, dtype=torch.float32)
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)

        if img_t.ndim == 3:
            last_frame = img_t.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        else:
            last_frame = img_t.to(device=self.device, dtype=self.dtype)

        vid = torch.zeros(
            1, 3, frames_per_step + 1, height, width, device=self.device, dtype=self.dtype
        )
        vid[0, :, 0] = last_frame[0]

        with torch.no_grad():
            latent = self._model.encode(
                (vid * 255.0).to(torch.uint8).to(dtype=self.dtype) / 255.0,
            )
        return last_frame, latent

    def create_initial_latent(
        self, *, height: int, width: int, frames_per_step: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Create a random initial ``(latent, last_frame)`` pair."""
        gen_device = "cpu" if self.device == "cpu" else self.device
        gen = torch.Generator(device=gen_device).manual_seed(seed)
        lh, lw = height // self.SPATIAL_FACTOR, width // self.SPATIAL_FACTOR
        latent_t = self._model.tokenizer.get_latent_num_frames(frames_per_step + 1)
        latent = torch.randn(
            1,
            self.LATENT_CH,
            latent_t,
            lh,
            lw,
            generator=gen,
            dtype=self.dtype,
            device=self.device,
        )
        last_frame = torch.zeros(1, 3, height, width, dtype=self.dtype, device=self.device)
        return latent, last_frame

    def create_initial_state(
        self,
        *,
        prompt: str,
        initial_image: Any | None,
        height: int,
        width: int,
        frames_per_step: int,
        seed: int,
    ) -> CosmosPredict2Latent:
        r"""Build a fresh ``CosmosPredict2Latent`` from a prompt and optional image."""
        text_emb = self.encode_text(prompt)
        if initial_image is not None:
            last_frame, latent = self.encode_image(
                initial_image, height=height, width=width, frames_per_step=frames_per_step
            )
        else:
            latent, last_frame = self.create_initial_latent(
                height=height, width=width, frames_per_step=frames_per_step, seed=seed
            )
        return CosmosPredict2Latent(latent, last_frame, text_emb, self._neg_text_emb)

    def denoise(
        self,
        state: CosmosPredict2Latent,
        *,
        num_steps: int = 35,
        guidance: float = 7.0,
        seed: int = 1,
        extras: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Run the full denoising loop. Returns ``(latent, video)`` (video in ``[-1, 1]``)."""
        data_batch = self._build_data_batch(state, extras=extras)

        with torch.no_grad():
            latent = self._model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                guidance=guidance,
                seed=seed,
                is_negative_prompt=state.neg_text_emb is not None,
                num_steps=num_steps,
            )
            video = self._model.decode(latent)
        return latent, video

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        r"""VAE-decode a latent tensor to a video tensor in ``[-1, 1]``."""
        with torch.no_grad():
            return self._model.decode(latent)

    def warmup(
        self,
        *,
        height: int,
        width: int,
        frames_per_step: int,
        extras: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        r"""Run a single-step denoise to warm caches and JIT-compile."""
        if not self._loaded:
            return
        state = self.create_initial_state(
            prompt="",
            initial_image=None,
            height=height,
            width=width,
            frames_per_step=frames_per_step,
            seed=0,
        )
        _ = self.denoise(state, num_steps=1, guidance=1.0, seed=0, extras=extras)

    def _build_data_batch(
        self,
        state: CosmosPredict2Latent,
        *,
        extras: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        last_frame = state.last_frame
        if last_frame.ndim == 3:
            last_frame = last_frame.unsqueeze(0)
        H, W = last_frame.shape[-2], last_frame.shape[-1]
        num_frames = self._model.tokenizer.get_pixel_num_frames(self._model.config.state_t)
        vid = torch.zeros(1, 3, num_frames, H, W, device=self.device, dtype=self.dtype)
        vid[0, :, 0] = last_frame[0]

        has_conditioning_frame = last_frame.abs().sum() > 0
        num_conditional = 1 if has_conditioning_frame else 0

        data_batch: dict[str, Any] = {
            "dataset_name": "video_data",
            "video": (vid.clamp(0, 1) * 255.0).to(torch.uint8),
            "fps": torch.tensor([24.0], device=self.device, dtype=self.dtype),
            "padding_mask": torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype),
            "num_conditional_frames": num_conditional,
            "t5_text_embeddings": state.text_emb,
        }
        if state.neg_text_emb is not None:
            data_batch["neg_t5_text_embeddings"] = state.neg_text_emb
        if extras:
            data_batch.update(extras)
        return data_batch

    def estimate_latent_vram_mb(self, *, height: int, width: int, frames_per_step: int) -> float:
        r"""Estimate per-session VRAM for latent + decode buffers (MB)."""
        lh, lw = height // self.SPATIAL_FACTOR, width // self.SPATIAL_FACTOR
        num_frames = frames_per_step + 1
        latent_bytes = self.LATENT_CH * ((num_frames + 3) // 4) * lh * lw * 2
        decode_bytes = 3 * height * width * num_frames * 4
        return (latent_bytes + decode_bytes) / (1024 * 1024) * 2.0 + 512.0

    # ---- VideoGenerator contract -----------------------------------------

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.encode_text(prompt)

    def initial_conditioning(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image: Any | None,
        height: int,
        width: int,
        frames_per_step: int,
        seed: int,
    ) -> CosmosPredict2Latent:
        return self.create_initial_state(
            prompt=prompt,
            initial_image=image,
            height=height,
            width=width,
            frames_per_step=frames_per_step,
            seed=seed,
        )

    def apply_prompt(
        self, conditioning: CosmosPredict2Latent, prompt_embeds: torch.Tensor
    ) -> CosmosPredict2Latent:
        return CosmosPredict2Latent(
            conditioning.latent, conditioning.last_frame, prompt_embeds, conditioning.neg_text_emb
        )

    def generate(
        self,
        conditioning: CosmosPredict2Latent,
        *,
        num_steps: int,
        guidance: float,
        num_frames: int,
        seed: int,
    ) -> GenerationResult:
        latent, video = self.denoise(
            conditioning, num_steps=num_steps, guidance=guidance, seed=seed
        )
        return GenerationResult(latent=latent, video=video)

    def advance(
        self, conditioning: CosmosPredict2Latent, next_image: torch.Tensor
    ) -> CosmosPredict2Latent:
        return CosmosPredict2Latent(
            conditioning.latent, next_image, conditioning.text_emb, conditioning.neg_text_emb
        )

    def profile_vram(self, *, height: int, width: int, num_frames: int) -> float:
        return self.estimate_latent_vram_mb(height=height, width=width, frames_per_step=num_frames)
