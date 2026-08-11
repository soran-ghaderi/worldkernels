r"""Shared weight-bearing nn.Modules for native video-diffusion models."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "RMSNorm",
    "SinusoidalTimesteps",
    "TimestepEmbedding",
    "PatchEmbed",
    "VideoRoPE3D",
    "apply_rotary_emb",
    "Attention",
    "FrameKVCache",
    "FeedForward",
    "DiTBlock",
    "FinalLayer",
    "VideoDiT",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RMSNorm": ("worldkernels.models.components.normalization", "RMSNorm"),
    "SinusoidalTimesteps": ("worldkernels.models.components.embeddings", "SinusoidalTimesteps"),
    "TimestepEmbedding": ("worldkernels.models.components.embeddings", "TimestepEmbedding"),
    "PatchEmbed": ("worldkernels.models.components.embeddings", "PatchEmbed"),
    "VideoRoPE3D": ("worldkernels.models.components.embeddings", "VideoRoPE3D"),
    "apply_rotary_emb": ("worldkernels.models.components.embeddings", "apply_rotary_emb"),
    "Attention": ("worldkernels.models.components.attention", "Attention"),
    "FrameKVCache": ("worldkernels.models.components.attention", "FrameKVCache"),
    "FeedForward": ("worldkernels.models.components.dit", "FeedForward"),
    "DiTBlock": ("worldkernels.models.components.dit", "DiTBlock"),
    "FinalLayer": ("worldkernels.models.components.dit", "FinalLayer"),
    "VideoDiT": ("worldkernels.models.components.dit", "VideoDiT"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
