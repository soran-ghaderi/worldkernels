r"""Attention backend selection.

Resolves a backend by explicit name or by the active platform's default,
falling back to SDPA when a requested accelerated backend is unavailable.
"""

from __future__ import annotations

from worldkernels.runtime.attention.backend import AttentionBackend

__all__ = ["select_attention_backend"]


def select_attention_backend(name: str | None = None) -> AttentionBackend:
    r"""Return an attention backend instance.

    Args:
        name: ``"sdpa"`` / ``"flash"``; ``None`` uses the platform default.
            A requested ``"flash"`` backend silently falls back to SDPA when
            ``flash-attn`` is not installed.
    """
    from worldkernels.platforms import current_platform
    from worldkernels.runtime.attention.backends import FlashAttentionBackend, SDPABackend

    resolved = name or current_platform().default_attention_backend()
    if resolved == "flash" and FlashAttentionBackend.is_available():
        return FlashAttentionBackend()
    if resolved == "flash":
        return SDPABackend()
    if resolved == "sdpa":
        return SDPABackend()
    raise ValueError(f"unknown attention backend {resolved!r}; available: sdpa, flash")
