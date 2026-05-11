r"""Capture golden tensors from the cosmos_predict2 wrapper path for native-rewrite validation.

Drives DreamDojoWorld (action-conditioned) or CosmosPredict2World (video-to-world)
end-to-end at a fixed seed and fixed inputs, snapshotting named tensors at every
numerical boundary the native rewrite must reproduce: text embedding, VAE
encode/decode, scheduler step, DiT block (first / middle / last), full DiT
forward, full-pipeline latent and decoded video. Outputs are written to
``{output_root}/{name}/`` as one ``.safetensors`` per tensor plus a
``manifest.json`` recording capture provenance (torch version, GPU, CUDA, seed).

Invocation:
    python -m tests.fixtures.cosmos_reference --adapter dreamdojo --variant 2b_pretrain
    pytest tests/native/ --regen-fixtures
"""

from __future__ import annotations

import argparse
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

FIXTURE_VERSION = 1
DEFAULT_SEED = 1234
DEFAULT_PROMPT = "A robot arm picks up a red block and places it on a wooden table."


@dataclass
class CaptureConfig:
    r"""Inputs that fully determine a fixture set."""

    adapter: str = "dreamdojo"
    variant: str = "2b_pretrain"
    height: int = 240
    width: int = 320
    pixel_frames: int = 5
    dtype_str: str = "bfloat16"
    device: str = "cuda"
    seed: int = DEFAULT_SEED
    prompt: str = DEFAULT_PROMPT
    num_steps: int = 4
    guidance: float = 3.0
    action_dim: int = 384
    chunk_size: int = 12
    output_root: Path = field(default_factory=lambda: Path("tests/fixtures/data"))
    dit_block_indices: tuple[int, ...] | None = None

    @property
    def name(self) -> str:
        return (
            f"{self.adapter}_{self.variant}"
            f"_{self.height}x{self.width}_{self.pixel_frames}f"
            f"_{self.dtype_str}_seed{self.seed}_steps{self.num_steps}"
        )

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.name


class _CaptureBuffer:
    def __init__(self) -> None:
        self.tensors: dict[str, "torch.Tensor"] = {}
        self.meta: dict[str, dict[str, Any]] = {}

    def record(self, name: str, t: "torch.Tensor", **meta: Any) -> None:
        if name in self.tensors:
            raise KeyError(f"capture name collision: {name!r} already recorded")
        cpu_t = t.detach().to("cpu", copy=True).contiguous()
        self.tensors[name] = cpu_t
        info: dict[str, Any] = {"dtype": str(cpu_t.dtype), "shape": list(cpu_t.shape)}
        info.update(meta)
        self.meta[name] = info


def _parse_dtype(s: str) -> "torch.dtype":
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def _make_init_frame(h: int, w: int) -> "torch.Tensor":
    r"""Deterministic 8-bar color pattern in [0, 1]; exercises VAE color range."""
    import torch

    bar_idx = (torch.arange(w) * 8 // w).view(1, 1, w)
    channel_bits = torch.arange(3).view(3, 1, 1)
    bits = ((bar_idx >> channel_bits) & 1).to(torch.float32)
    return bits.expand(3, h, w).contiguous()


def _deterministic_joints(chunk_size: int, action_dim: int, seed: int) -> list[list[float]]:
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x9E3779B1)
    return (torch.rand(chunk_size, action_dim, generator=g) * 2.0 - 1.0).tolist()


@contextmanager
def _hook_dit_blocks(net, buf: _CaptureBuffer, block_indices: tuple[int, ...]) -> Iterator[None]:
    r"""Capture (input, output) for selected blocks on their FIRST forward call only.

    Under CFG (guidance > 1) the DiT runs twice per sampler step (cond + uncond).
    We pin to the first call so the fixture is unambiguous; native rewrite must
    reproduce the same call.
    """
    handles = []
    blocks = list(getattr(net, "blocks", []))
    seen_pre: set[int] = set()
    seen_post: set[int] = set()

    def _make_pre(idx: int):
        def _hook(_module, args, _kwargs):
            if idx in seen_pre or not args:
                return
            seen_pre.add(idx)
            buf.record(f"dit_block_{idx}_input_x", args[0])

        return _hook

    def _make_post(idx: int):
        def _hook(_module, _args, output):
            if idx in seen_post:
                return
            out_t = output if hasattr(output, "shape") else (output[0] if output else None)
            if out_t is None:
                return
            seen_post.add(idx)
            buf.record(f"dit_block_{idx}_output", out_t)

        return _hook

    for idx in block_indices:
        if idx >= len(blocks):
            continue
        b = blocks[idx]
        handles.append(b.register_forward_pre_hook(_make_pre(idx), with_kwargs=True))
        handles.append(b.register_forward_hook(_make_post(idx)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def _hook_dit_forward(net, buf: _CaptureBuffer) -> Iterator[None]:
    captured = False

    def _pre(_module, args, _kwargs):
        nonlocal captured
        if captured or not args:
            return
        buf.record("dit_forward_input_x", args[0])

    def _post(_module, _args, output):
        nonlocal captured
        if captured:
            return
        out_t = output if hasattr(output, "shape") else (output[0] if output else None)
        if out_t is None:
            return
        buf.record("dit_forward_output", out_t)
        captured = True

    h_pre = net.register_forward_pre_hook(_pre, with_kwargs=True)
    h_post = net.register_forward_hook(_post)
    try:
        yield
    finally:
        h_pre.remove()
        h_post.remove()


@contextmanager
def _patch_scheduler_step(model, buf: _CaptureBuffer) -> Iterator[None]:
    sched = getattr(model, "sample_scheduler", None)
    if sched is None or not hasattr(sched, "step"):
        yield
        return

    orig = sched.step
    captured = False

    def _wrapped(*args, **kwargs):
        nonlocal captured
        if captured:
            return orig(*args, **kwargs)
        import torch

        pos = list(args)
        model_output = pos[0] if pos else kwargs.get("model_output")
        timestep = pos[1] if len(pos) > 1 else kwargs.get("timestep")
        sample = pos[2] if len(pos) > 2 else kwargs.get("sample")
        if model_output is not None:
            buf.record("sampler_step_eps_pred", model_output)
        if sample is not None:
            buf.record("sampler_step_x_t", sample)
        if timestep is not None:
            buf.record("sampler_step_t", torch.as_tensor(timestep))
        out = orig(*args, **kwargs)
        next_sample = getattr(out, "prev_sample", out)
        if hasattr(next_sample, "shape"):
            buf.record("sampler_step_x_t_next", next_sample)
        captured = True
        return out

    sched.step = _wrapped
    try:
        yield
    finally:
        sched.step = orig


@contextmanager
def _patch_vae_decode(model, buf: _CaptureBuffer) -> Iterator[None]:
    orig = model.decode
    captured = False

    def _wrapped(latent, *args, **kwargs):
        nonlocal captured
        out = orig(latent, *args, **kwargs)
        if not captured:
            buf.record("pipeline_vae_decode_input", latent)
            buf.record("pipeline_decoded_float", out)
            captured = True
        return out

    model.decode = _wrapped
    try:
        yield
    finally:
        model.decode = orig


def capture(cfg: CaptureConfig) -> Path:
    r"""Run the wrapper end-to-end at fixed inputs, snapshot every numerical boundary."""
    import torch

    from worldkernels.core.action import Action
    from worldkernels.core.config import WorldConfig
    from worldkernels.worlds.adapters._cosmos_predict2 import CosmosLatent
    from worldkernels.worlds.adapters._cosmos_predict2._deps import ensure_cosmos_predict2

    ensure_cosmos_predict2()
    dtype = _parse_dtype(cfg.dtype_str)

    if cfg.adapter == "dreamdojo":
        from worldkernels.worlds.adapters.dreamdojo import DreamDojoWorld

        world = DreamDojoWorld(
            variant=cfg.variant,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.guidance,
        )
        is_action_conditioned = True
    elif cfg.adapter == "cosmos":
        from worldkernels.worlds.adapters.cosmos import CosmosPredict2World

        world = CosmosPredict2World(
            num_inference_steps=cfg.num_steps,
            guidance_scale=cfg.guidance,
        )
        is_action_conditioned = False
    else:
        raise ValueError(f"unknown adapter {cfg.adapter!r}")

    log.info("Initializing %s/%s on %s/%s", cfg.adapter, cfg.variant, cfg.device, dtype)
    world.initialize(cfg.device, dtype)
    buf = _CaptureBuffer()

    text_emb = world._compute_text_embedding(cfg.prompt)
    buf.record("text_emb", text_emb, prompt=cfg.prompt)
    if world._neg_text_emb is not None:
        buf.record("neg_text_emb", world._neg_text_emb)

    init_frame = _make_init_frame(cfg.height, cfg.width).to(cfg.device, dtype)
    pixel_frames_for_encode = cfg.pixel_frames + 1
    vid = torch.zeros(
        1, 3, pixel_frames_for_encode, cfg.height, cfg.width, dtype=dtype, device=cfg.device
    )
    vid[0, :, 0] = init_frame
    enc_input = (vid * 255.0).to(torch.uint8).to(dtype) / 255.0
    buf.record("vae_encode_input", enc_input)
    with torch.no_grad():
        enc_output = world._model.encode(enc_input)
    buf.record("vae_encode_output", enc_output)

    buf.record("vae_decode_input", enc_output)
    with torch.no_grad():
        dec_output = world._model.decode(enc_output)
    buf.record("vae_decode_output", dec_output)

    wcfg = WorldConfig(
        height=cfg.height,
        width=cfg.width,
        frames_per_step=cfg.pixel_frames,
        initial_prompt=cfg.prompt,
    )
    state = world.create_initial_state(wcfg, seed=cfg.seed)
    buf.record("pipeline_init_noise", state.data.latent)
    buf.record("pipeline_text_emb", state.data.text_emb)

    if is_action_conditioned:
        joints = _deterministic_joints(cfg.chunk_size, cfg.action_dim, cfg.seed)
        action = Action("continuous", {"joints": joints})
        action_encoded = world.encode_action(action)
        buf.record("pipeline_action", action_encoded)
    else:
        action_encoded = torch.empty(0, device=cfg.device, dtype=dtype)

    resolved_text_emb = world._resolve_text_emb(state.data, action_encoded)
    cs_for_diffusion = CosmosLatent(
        latent=state.data.latent,
        last_frame=state.data.last_frame,
        text_emb=resolved_text_emb,
        neg_text_emb=state.data.neg_text_emb,
    )

    net = world._model.net
    blocks = list(getattr(net, "blocks", []))
    if not blocks:
        raise RuntimeError(f"adapter {cfg.adapter!r} has no .net.blocks; capture flow needs update")
    n_blocks = len(blocks)
    block_indices = cfg.dit_block_indices or (0, n_blocks // 2, n_blocks - 1)
    log.info("DiT has %d blocks; hooking %s", n_blocks, block_indices)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    with ExitStack() as stack:
        stack.enter_context(_hook_dit_blocks(net, buf, block_indices))
        stack.enter_context(_hook_dit_forward(net, buf))
        stack.enter_context(_patch_scheduler_step(world._model, buf))
        stack.enter_context(_patch_vae_decode(world._model, buf))
        diffusion_action = action_encoded if action_encoded.numel() > 0 else None
        new_latent, _ = world._run_diffusion(
            cs_for_diffusion,
            num_steps=cfg.num_steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
            action_encoded=diffusion_action,
        )

    buf.record("pipeline_final_latent", new_latent)

    decoded_float = buf.tensors.get("pipeline_decoded_float")
    if decoded_float is not None:
        decoded_uint8 = ((decoded_float + 1.0) * 0.5).clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        buf.record("pipeline_decoded_uint8", decoded_uint8)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_safetensors(out_dir, buf.tensors)
    _write_manifest(out_dir, cfg, buf.meta, tuple(block_indices), n_blocks)
    log.info("Captured %d tensors to %s", len(buf.tensors), out_dir)
    return out_dir


def _write_safetensors(out_dir: Path, tensors: dict[str, "torch.Tensor"]) -> None:
    from safetensors.torch import save_file

    for name, t in tensors.items():
        save_file({"tensor": t}, str(out_dir / f"{name}.safetensors"))


def _write_manifest(
    out_dir: Path,
    cfg: CaptureConfig,
    tensor_meta: dict[str, dict[str, Any]],
    block_indices: tuple[int, ...],
    n_blocks: int,
) -> None:
    import torch

    cfg_d = asdict(cfg)
    cfg_d["output_root"] = str(cfg.output_root)
    cfg_d["dit_block_indices"] = list(cfg.dit_block_indices) if cfg.dit_block_indices else None

    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "captured_at_unix": int(time.time()),
        "name": cfg.name,
        "config": cfg_d,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_version": torch.version.cuda,
        "dit_blocks_total": n_blocks,
        "dit_blocks_hooked": list(block_indices),
        "tensors": tensor_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _parse_args(argv: list[str]) -> CaptureConfig:
    p = argparse.ArgumentParser(prog="cosmos_reference", description=__doc__)
    p.add_argument("--adapter", default="dreamdojo", choices=("dreamdojo", "cosmos"))
    p.add_argument("--variant", default="2b_pretrain")
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--pixel-frames", type=int, default=5)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--action-dim", type=int, default=384)
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--output-root", default="tests/fixtures/data")
    ns = p.parse_args(argv)
    return CaptureConfig(
        adapter=ns.adapter,
        variant=ns.variant,
        height=ns.height,
        width=ns.width,
        pixel_frames=ns.pixel_frames,
        dtype_str=ns.dtype,
        device=ns.device,
        seed=ns.seed,
        prompt=ns.prompt,
        num_steps=ns.num_steps,
        guidance=ns.guidance,
        action_dim=ns.action_dim,
        chunk_size=ns.chunk_size,
        output_root=Path(ns.output_root),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])
    out = capture(cfg)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
