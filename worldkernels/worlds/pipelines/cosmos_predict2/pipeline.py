r"""Video diffusion pipeline backed by NVIDIA's cosmos_predict2 package.

Wraps cosmos_predict2 for inference: model loading + path patching, text/image
encoding, the denoising loop, and VAE decode. Adapters compose this pipeline
rather than inheriting from it — the AbstractWorld 3-stage contract lives on
the adapter, the pipeline handles all cosmos_predict2-specific machinery.

Phase 5 replaces this with a native pipeline under worldkernels/pipelines/.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


def _download_hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


class CosmosPredict2Latent:
    r"""Latent state for cosmos_predict2-backed pipelines."""

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


class CosmosPredict2Pipeline:
    r"""Video diffusion pipeline driving cosmos_predict2 for inference.

    Composed by adapters (cosmos, dreamdojo). The pipeline owns model loading,
    text/image encoding, denoising, and VAE decode. Adapters own action encoding,
    stage-level orchestration, and per-step extras (e.g. DreamDojo joint vectors).
    """

    LATENT_CH: int = 16
    SPATIAL_FACTOR: int = 8
    HF_TOKENIZER_REPO: str = "nvidia/Cosmos-Predict2.5-2B"
    NEGATIVE_PROMPT: str = (
        "The video captures a series of frames showing ugly scenes, static with no motion, "
        "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
        "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
        "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
        "artifacting, color banding, unnatural transitions, outdated special effects, fake "
        "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, "
        "and flickering. Overall, the video is of poor quality."
    )

    def __init__(self, *, experiment: str, config_file: str) -> None:
        self.experiment = experiment
        self.config_file = config_file
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._model: Any = None
        self._neg_text_emb: torch.Tensor | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def neg_text_emb(self) -> torch.Tensor | None:
        return self._neg_text_emb

    def load(self, device: str, dtype: torch.dtype, ckpt_path: str) -> None:
        r"""Load model weights and prepare the pipeline for inference."""
        from worldkernels.worlds.pipelines.cosmos_predict2.deps import ensure_cosmos_predict2

        ensure_cosmos_predict2()

        self.device = device
        self.dtype = dtype

        log.info("Loading model (experiment=%s)", self.experiment)
        self._model = self._load_model(self.experiment, ckpt_path, self.config_file)
        self._neg_text_emb = self.encode_text(self.NEGATIVE_PROMPT)
        self._loaded = True
        log.info(
            "CosmosPredict2 pipeline loaded on %s (%.1f GB VRAM)",
            device,
            torch.cuda.memory_allocated() / 1e9,
        )

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

        self._patch_tokenizer_path()
        self._patch_text_encoder_paths()

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
        try:
            tokenizer_path = _download_hf_file(self.HF_TOKENIZER_REPO, "tokenizer.pth")
        except Exception:
            log.warning("Could not download tokenizer from HF")
            return

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

    def encode_text(self, prompt: str) -> torch.Tensor:
        r"""Compute text embedding for ``prompt``."""
        if self._model is not None and getattr(self._model, "text_encoder", None) is not None:
            emb = self._model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
            return emb.to(device=self.device, dtype=self.dtype)
        from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding

        return get_text_embedding(prompt).to(device=self.device, dtype=self.dtype)

    def encode_image(
        self, image: Any, *, height: int, width: int, frames_per_step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Encode an initial image into (last_frame, latent)."""
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
        r"""Create a random initial (latent, last_frame) pair."""
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
        r"""Build a fresh ``CosmosPredict2Latent`` from prompt and optional image."""
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
        r"""Run the full denoising loop. Returns ``(new_latent, new_last_frame)``."""
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

        last_frame = ((video + 1.0) * 0.5)[0, :, -1].clamp(0, 1)
        return latent, last_frame

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
        r"""Run a single-step denoise to warm caches and JIT compile."""
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
        last_frame = (
            state.last_frame.unsqueeze(0) if state.last_frame.ndim == 3 else state.last_frame
        )
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
