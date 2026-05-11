r"""Per-stage tolerance contract and drift-reporting assertion for the native rewrite.

A stage's `out` is compared against the captured reference under
$$ |\text{out} - \text{ref}| \le \text{atol} + \text{rtol} \cdot |\text{ref}| $$
element-wise. On failure the report names the worst-over-budget element, its
index, max-abs / mean-abs / max-rel drift, and the active tolerance budget.

`TOLERANCES` keys are tolerance *classes* (e.g. ``vae_encode`` applies to any
tensor produced by the VAE encoder), not fixture *file* names. The same class
can be reused across multiple fixtures (input/output pairs, multiple blocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "StageTolerance",
    "TOLERANCES",
    "DriftReport",
    "assert_close_to_reference",
    "load_reference",
]


@dataclass(frozen=True)
class StageTolerance:
    atol: float
    rtol: float
    description: str


TOLERANCES: dict[str, StageTolerance] = {
    "text_embedding": StageTolerance(
        1e-4, 1e-4, "Qwen2.5-VL-7B output; kept via HF transformers, not rewritten"
    ),
    "vae_encode": StageTolerance(
        5e-3, 5e-3, "WanVAE Encoder3d on a single-frame video; bf16 baseline"
    ),
    "vae_decode": StageTolerance(
        5e-3, 5e-3, "WanVAE Decoder3d on an encoded latent; bf16 baseline"
    ),
    "sampler_step": StageTolerance(
        1e-6,
        1e-6,
        "FlowUniPCMultistepScheduler.step is closed-form given (x_t, t, eps); near machine precision",
    ),
    "dit_block": StageTolerance(
        1e-2, 1e-2, "A single WanAttentionBlock forward; bf16, accumulates rounding within block"
    ),
    "dit_forward": StageTolerance(
        3e-2,
        3e-2,
        "Full WanModel forward; drift compounds across ~30 blocks for 2B / ~40 for 14B",
    ),
    "pipeline_latent": StageTolerance(
        5e-3, 5e-3, "End-to-end denoised latent vs reference; sampler+DiT compound budget"
    ),
    "pipeline_decoded_float": StageTolerance(
        5e-3, 5e-3, "VAE-decoded video in [-1, 1] float space; before uint8 quantization"
    ),
    "pipeline_decoded_uint8": StageTolerance(
        2.0, 0.0, "Decoded video in uint8 space; max-abs diff of <= 2 LSB (user-visible budget)"
    ),
}


@dataclass
class DriftReport:
    stage: str
    name: str
    max_abs: float
    mean_abs: float
    max_rel: float
    worst_idx: tuple[int, ...]
    worst_overshoot: float
    n_over_budget: int
    n_total: int
    ref_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    ref_dtype: str
    out_dtype: str
    tolerance: StageTolerance
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status} {self.stage}:{self.name}]\n"
            f"  max_abs={self.max_abs:.4e}  mean_abs={self.mean_abs:.4e}  max_rel={self.max_rel:.4e}\n"
            f"  worst_idx={self.worst_idx}  overshoot={self.worst_overshoot:.4e}\n"
            f"  over_budget={self.n_over_budget}/{self.n_total}\n"
            f"  ref shape={self.ref_shape} dtype={self.ref_dtype}\n"
            f"  out shape={self.out_shape} dtype={self.out_dtype}\n"
            f"  budget atol={self.tolerance.atol:g} rtol={self.tolerance.rtol:g}"
        )


def assert_close_to_reference(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    stage: str,
    name: str = "",
) -> DriftReport:
    r"""Compare `out` against `ref` under the `stage` tolerance, raising on failure.

    Args:
        out: Tensor produced by the native rewrite.
        ref: Golden reference tensor (loaded via `load_reference`).
        stage: Key into `TOLERANCES`.
        name: Free-form label for the report (e.g. block index, step index).

    Returns:
        DriftReport. Caller may log it on success (e.g. CI drift tracking).
    """
    import torch

    if stage not in TOLERANCES:
        raise KeyError(f"unknown tolerance stage {stage!r}; declared: {sorted(TOLERANCES)}")
    tol = TOLERANCES[stage]

    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(
            f"[{stage}:{name}] shape mismatch: out={tuple(out.shape)} ref={tuple(ref.shape)}"
        )

    out_f = out.detach().to(torch.float32)
    ref_f = ref.detach().to(torch.float32)
    diff = (out_f - ref_f).abs()
    budget = tol.atol + tol.rtol * ref_f.abs()
    over = diff - budget
    n_over = int((over > 0).sum().item())

    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rel = diff / (ref_f.abs() + 1e-12)
    max_rel = float(rel.max().item())

    worst_overshoot = float(over.max().item())
    worst_idx = tuple(int(i) for i in torch.unravel_index(over.flatten().argmax(), out_f.shape))

    report = DriftReport(
        stage=stage,
        name=name or stage,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        worst_idx=worst_idx,
        worst_overshoot=worst_overshoot,
        n_over_budget=n_over,
        n_total=int(diff.numel()),
        ref_shape=tuple(ref.shape),
        out_shape=tuple(out.shape),
        ref_dtype=str(ref.dtype),
        out_dtype=str(out.dtype),
        tolerance=tol,
        passed=(n_over == 0),
    )

    if not report.passed:
        raise AssertionError(str(report))
    return report


def load_reference(fixture_dir: Path, name: str) -> torch.Tensor:
    r"""Load a captured reference tensor by name from `fixture_dir`."""
    from safetensors.torch import load_file

    path = fixture_dir / f"{name}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"reference fixture not found: {path}")
    data = load_file(str(path))
    if "tensor" not in data:
        raise KeyError(f"fixture {path} missing 'tensor' key (found {sorted(data)})")
    return data["tensor"]
