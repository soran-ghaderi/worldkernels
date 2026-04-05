r"""Shared base for cosmos_predict2-based world model adapters.

Both CosmosPredict2World (video-to-world) and DreamDojoWorld (action-conditioned)
share model loading, tokenizer/text-encoder patching, VAE decode, and diffusion
sampling. This base class provides all of that; subclasses only implement
encode_action and optionally override _build_data_batch.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import AbstractWorld

if TYPE_CHECKING:
    from worldkernels.core.config import WorldConfig

log = logging.getLogger(__name__)

_LATENT_CH = 16
_SPATIAL_FACTOR = 8

_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)

_HF_COSMOS_REPO = "nvidia/Cosmos-Predict2.5-2B"
_HF_DREAMDOJO_REPO = "nvidia/DreamDojo"


class CosmosLatent:
    r"""Structured latent state shared by all cosmos_predict2-based adapters."""

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

    def clone(self) -> CosmosLatent:
        return CosmosLatent(
            latent=self.latent.clone(),
            last_frame=self.last_frame.clone(),
            text_emb=self.text_emb.clone(),
            neg_text_emb=self.neg_text_emb.clone() if self.neg_text_emb is not None else None,
        )

    def to(self, device: Any) -> CosmosLatent:
        return CosmosLatent(
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


def download_hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")


def download_dreamdojo_checkpoint(ckpt_dir_name: str = "2B_pretrain") -> str:
    r"""Download DreamDojo checkpoint (DCP format) and convert to .pt."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        _HF_DREAMDOJO_REPO,
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


class CosmosBaseWorld(AbstractWorld):
    r"""Base for all cosmos_predict2-backed world models.

    Subclasses must set ``name``, ``config_file``, ``default_experiment``, and
    implement ``encode_action``. Override ``_build_data_batch`` if the model
    needs extra conditioning (e.g. action tensors for DreamDojo).
    """

    name: str = ""
    config_file: str = ""
    default_experiment: str = ""
    hf_repo: str = _HF_COSMOS_REPO

    stage_exec_modes = {
        StageType.ENCODE: StageExecMode.SINGLE_SHOT,
        StageType.TRANSITION: StageExecMode.ITERATIVE,
        StageType.DECODE: StageExecMode.SINGLE_SHOT,
    }
    transition_mode = TransitionMode.BIDIRECTIONAL
    supports_streaming = False
    supports_kv_cache = False

    def __init__(
        self,
        ckpt_path: str | None = None,
        experiment: str | None = None,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        **kwargs: Any,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.experiment = experiment or self.default_experiment
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._model: Any = None
        self._neg_text_emb: torch.Tensor | None = None
        self._initialized = False

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        from worldkernels.worlds.adapters._cosmos_deps import ensure_cosmos_predict2

        ensure_cosmos_predict2()

        self.device = device
        self.dtype = dtype

        ckpt = self._resolve_checkpoint()

        log.info("Loading model (experiment=%s)", self.experiment)
        self._model = self._load_model(self.experiment, ckpt, self.config_file)
        self._neg_text_emb = self._compute_text_embedding(_DEFAULT_NEGATIVE_PROMPT)
        self._initialized = True
        log.info(
            "%s initialized on %s (%.1f GB VRAM)",
            self.name,
            device,
            torch.cuda.memory_allocated() / 1e9,
        )

    def _resolve_checkpoint(self) -> str:
        r"""Return local path to model checkpoint, downloading if needed."""
        raise NotImplementedError

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
            tokenizer_path = download_hf_file(_HF_COSMOS_REPO, "tokenizer.pth")
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

    def warmup(self, config: WorldConfig) -> None:
        if not self._initialized:
            return
        state = self.create_initial_state(config, seed=0)
        _ = self._run_diffusion(state.data, num_steps=1, guidance=1.0, seed=0)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        cs: CosmosLatent = state.data
        text_emb = self._resolve_text_emb(cs, action_encoded)

        new_latent, new_last_frame = self._run_diffusion(
            CosmosLatent(cs.latent, cs.last_frame, text_emb, cs.neg_text_emb),
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
            seed=int(time.perf_counter() * 1000) % (2**31),
            action_encoded=action_encoded,
        )
        return LatentState(
            data=CosmosLatent(new_latent, new_last_frame, text_emb, cs.neg_text_emb),
            device=state.device,
        )

    def _resolve_text_emb(self, cs: CosmosLatent, action_encoded: torch.Tensor) -> torch.Tensor:
        r"""Determine text embedding for this step. Override in subclasses where
        action_encoded is NOT a text embedding (e.g. DreamDojo joint vectors)."""
        return action_encoded if action_encoded.numel() > 0 else cs.text_emb

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        t0 = time.perf_counter()
        cs: CosmosLatent = state.data
        frames = None
        latent_out = None

        if "frames" in modalities:
            with torch.no_grad():
                video = self._model.decode(cs.latent)
            video_uint8 = ((video + 1.0) * 0.5).clamp(0, 1).mul(255).to(torch.uint8)
            frames = [
                video_uint8[0, :, t].permute(1, 2, 0).cpu().numpy().tobytes()
                for t in range(video_uint8.shape[2])
            ]

        if "latent" in modalities:
            latent_out = cs.latent

        return Observation(
            step_index=0,
            generation_time_ms=(time.perf_counter() - t0) * 1000.0,
            frames=frames,
            latent=latent_out,
        )

    def estimate_vram_mb(self, config: WorldConfig) -> float:
        lh, lw = config.height // _SPATIAL_FACTOR, config.width // _SPATIAL_FACTOR
        num_frames = config.frames_per_step + 1
        latent_bytes = _LATENT_CH * ((num_frames + 3) // 4) * lh * lw * 2
        decode_bytes = 3 * config.height * config.width * num_frames * 4
        return (latent_bytes + decode_bytes) / (1024 * 1024) * 2.0 + 512.0

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        text_emb = self._compute_text_embedding(config.initial_prompt or "")

        if config.initial_image is not None:
            last_frame, latent = self._encode_initial_image(config)
        else:
            gen_device = "cpu" if self.device == "cpu" else self.device
            gen = torch.Generator(device=gen_device).manual_seed(seed)
            lh, lw = config.height // _SPATIAL_FACTOR, config.width // _SPATIAL_FACTOR
            latent_t = self._model.tokenizer.get_latent_num_frames(config.frames_per_step + 1)
            latent = torch.randn(
                1,
                _LATENT_CH,
                latent_t,
                lh,
                lw,
                generator=gen,
                dtype=self.dtype,
                device=self.device,
            )
            last_frame = torch.zeros(
                1,
                3,
                config.height,
                config.width,
                dtype=self.dtype,
                device=self.device,
            )

        return LatentState(
            data=CosmosLatent(latent, last_frame, text_emb, self._neg_text_emb),
            device=self.device,
        )

    def _compute_text_embedding(self, prompt: str) -> torch.Tensor:
        if self._model is not None and getattr(self._model, "text_encoder", None) is not None:
            emb = self._model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
            return emb.to(device=self.device, dtype=self.dtype)
        from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding

        return get_text_embedding(prompt).to(device=self.device, dtype=self.dtype)

    def _encode_initial_image(self, config: WorldConfig) -> tuple[torch.Tensor, torch.Tensor]:
        import torchvision.transforms.functional as TF

        if isinstance(config.initial_image, str):
            from PIL import Image

            img = Image.open(config.initial_image).convert("RGB")
            img_t = TF.to_tensor(img.resize((config.width, config.height)))
        else:
            img_t = torch.as_tensor(config.initial_image, dtype=torch.float32)
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)

        if img_t.ndim == 3:
            last_frame = img_t.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0)
        else:
            last_frame = img_t.to(
                device=self.device,
                dtype=self.dtype,
            )
        vid = torch.zeros(
            1,
            3,
            config.frames_per_step + 1,
            config.height,
            config.width,
            device=self.device,
            dtype=self.dtype,
        )
        vid[0, :, 0] = last_frame[0]

        with torch.no_grad():
            latent = self._model.encode(
                (vid * 255.0).to(torch.uint8).to(dtype=self.dtype) / 255.0,
            )
        return last_frame, latent

    def _build_data_batch(
        self,
        cs: CosmosLatent,
        action_encoded: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        last_frame = cs.last_frame.unsqueeze(0) if cs.last_frame.ndim == 3 else cs.last_frame
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
            "t5_text_embeddings": cs.text_emb,
        }
        if cs.neg_text_emb is not None:
            data_batch["neg_t5_text_embeddings"] = cs.neg_text_emb
        return data_batch

    def _run_diffusion(
        self,
        cs: CosmosLatent,
        num_steps: int = 35,
        guidance: float = 7.0,
        seed: int = 1,
        action_encoded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_batch = self._build_data_batch(cs, action_encoded=action_encoded)

        with torch.no_grad():
            latent = self._model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                guidance=guidance,
                seed=seed,
                is_negative_prompt=cs.neg_text_emb is not None,
                num_steps=num_steps,
            )
            video = self._model.decode(latent)

        last_frame = ((video + 1.0) * 0.5)[0, :, -1].clamp(0, 1)
        return latent, last_frame
