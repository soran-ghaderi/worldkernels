r"""Capture golden tensors from the native DreamDojo pipeline for regression tests.

Runs `DreamDojoPipeline.generate_clip` at fixed
inputs and snapshots every numerical boundary: VAE encode/decode, first
scheduler step, DiT block (first/mid/last) and full-forward tensors, final
latent, and the decoded uint8 video. The captured set was verified bit-identical
to the pre-deletion ``cosmos_predict2`` wrapper at the same seeds.

Invocation:
    python -m tests.fixtures.dreamdojo_reference
    pytest tests/native/ --regen-fixtures
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

FIXTURE_VERSION = 2
DEFAULT_SEED = 1234


@dataclass
class CaptureConfig:
    r"""Inputs that fully determine a fixture set."""

    variant: str = "2b_pretrain"
    height: int = 240
    width: int = 320
    dtype_str: str = "bfloat16"
    device: str = "cuda"
    seed: int = DEFAULT_SEED
    num_steps: int = 4
    guidance: float = 0.0
    action_dim: int = 384
    chunk_size: int = 12
    ckpt_path: str | None = None
    output_root: Path = field(default_factory=lambda: Path("tests/fixtures/data"))

    @property
    def name(self) -> str:
        return (
            f"dreamdojo_{self.variant}_{self.height}x{self.width}"
            f"_{self.dtype_str}_seed{self.seed}_steps{self.num_steps}_g{self.guidance:g}"
        )

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.name


class _CaptureBuffer:
    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.meta: dict[str, dict[str, Any]] = {}

    def record(self, name: str, t: "torch.Tensor", **meta: Any) -> None:
        if name in self.tensors:
            raise KeyError(f"capture name collision: {name!r}")
        cpu_t = t.detach().to("cpu", copy=True).contiguous()
        self.tensors[name] = cpu_t
        self.meta[name] = {"dtype": str(cpu_t.dtype), "shape": list(cpu_t.shape), **meta}


def make_cond_frame(h: int, w: int) -> "torch.Tensor":
    r"""Deterministic 8-bar color pattern, ``(1, 3, 1, H, W)`` uint8."""
    import torch

    bar_idx = (torch.arange(w) * 8 // w).view(1, 1, w)
    channel_bits = torch.arange(3).view(3, 1, 1)
    bits = ((bar_idx >> channel_bits) & 1).to(torch.float32)
    frame = bits.expand(3, h, w).contiguous()
    return (frame * 255.0).to(torch.uint8).unsqueeze(0).unsqueeze(2)


def make_action(chunk_size: int, action_dim: int, seed: int) -> "torch.Tensor":
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x9E3779B1)
    return torch.rand(1, chunk_size, action_dim, generator=g) * 2.0 - 1.0


@contextmanager
def _hook_blocks(net, buf: _CaptureBuffer) -> Iterator[None]:
    handles = []
    n = len(net.blocks)
    indices = (0, n // 2, n - 1)
    seen: set[str] = set()

    def _make(idx: int, kind: str):
        def _pre(_m, args, _kw):
            key = f"dit_block_{idx}_input_x"
            if key in seen or not args:
                return
            seen.add(key)
            buf.record(key, args[0])

        def _post(_m, _args, output):
            key = f"dit_block_{idx}_output"
            if key in seen:
                return
            seen.add(key)
            buf.record(key, output)

        return _pre if kind == "pre" else _post

    for idx in indices:
        handles.append(
            net.blocks[idx].register_forward_pre_hook(_make(idx, "pre"), with_kwargs=True)
        )
        handles.append(net.blocks[idx].register_forward_hook(_make(idx, "post")))

    def _net_pre(_m, args, _kw):
        if "dit_forward_input_x" not in seen and args:
            seen.add("dit_forward_input_x")
            buf.record("dit_forward_input_x", args[0])

    def _net_post(_m, _args, output):
        if "dit_forward_output" not in seen:
            seen.add("dit_forward_output")
            buf.record("dit_forward_output", output)

    handles.append(net.register_forward_pre_hook(_net_pre, with_kwargs=True))
    handles.append(net.register_forward_hook(_net_post))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def capture(cfg: CaptureConfig) -> Path:
    r"""Run the native pipeline at fixed inputs, snapshot every numerical boundary."""
    import torch

    from worldkernels.models.dreamdojo.pipeline import DreamDojoPipeline
    from worldkernels.models.schedulers.flow_unipc import FlowUniPCMultistepScheduler
    from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype_str]
    pipeline = DreamDojoPipeline(cfg.variant, num_steps=cfg.num_steps, guidance=cfg.guidance).load(
        cfg.device, dtype, cfg.ckpt_path
    )
    buf = _CaptureBuffer()

    cond = make_cond_frame(cfg.height, cfg.width).to(cfg.device)
    action = make_action(cfg.chunk_size, cfg.action_dim, cfg.seed).to(cfg.device)
    buf.record("cond_frame_uint8", cond)
    buf.record("pipeline_action", action)
    assert pipeline.text_emb is not None
    buf.record("text_emb", pipeline.text_emb)

    vae_in = cond.to(dtype) / 127.5 - 1.0
    assert pipeline.tokenizer is not None
    with torch.no_grad():
        enc = pipeline.tokenizer.encode(vae_in)
        dec = pipeline.tokenizer.decode(enc)
    buf.record("vae_encode_input", vae_in)
    buf.record("vae_encode_output", enc)
    buf.record("vae_decode_output", dec)

    orig_step = FlowUniPCMultistepScheduler.step
    step_seen = False

    def _step(self, model_output, timestep, sample, **kw):
        nonlocal step_seen
        out = orig_step(self, model_output, timestep, sample, **kw)
        if not step_seen:
            step_seen = True
            buf.record("sampler_step_eps_pred", model_output)
            buf.record("sampler_step_x_t", sample)
            buf.record("sampler_step_t", torch.as_tensor(timestep))
            buf.record(
                "sampler_step_x_t_next", out[0] if isinstance(out, tuple) else out.prev_sample
            )
        return out

    with ExitStack() as stack:
        stack.enter_context(set_forward_context(ForwardContext(attention_backend="sdpa")))
        stack.enter_context(_hook_blocks(pipeline.net, buf))
        FlowUniPCMultistepScheduler.step = _step  # type: ignore[method-assign]
        try:
            latent, video = pipeline.generate_clip(
                cond, action, num_steps=cfg.num_steps, guidance=cfg.guidance, seed=cfg.seed
            )
        finally:
            FlowUniPCMultistepScheduler.step = orig_step  # type: ignore[method-assign]

    buf.record("pipeline_final_latent", latent)
    buf.record("pipeline_decoded_float", video)
    buf.record("pipeline_decoded_uint8", DreamDojoPipeline.to_uint8(video))

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    for name, t in buf.tensors.items():
        save_file({"tensor": t}, str(out_dir / f"{name}.safetensors"))
    cfg_d = asdict(cfg)
    cfg_d["output_root"] = str(cfg.output_root)
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "captured_at_unix": int(time.time()),
        "name": cfg.name,
        "config": cfg_d,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tensors": buf.meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("captured %d tensors to %s", len(buf.tensors), out_dir)
    print(out_dir)
    return out_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    capture(CaptureConfig())
    capture(CaptureConfig(guidance=3.0, seed=5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
