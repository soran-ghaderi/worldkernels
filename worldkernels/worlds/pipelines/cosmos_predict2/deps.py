r"""Dependency setup for cosmos_predict2 integration.

Handles three concerns:
1. Inject lightweight stubs for training-only deps (megatron, transformer_engine)
   so the inference import chain does not fail.
2. Locate the cosmos_predict2 package (local DreamDojo checkout or pip-installed)
   and add it to sys.path if needed.
3. Validate that the package is actually importable.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from pathlib import Path

log = logging.getLogger(__name__)

_setup_done = False


def _inject_stub(module_name: str, attrs: dict | None = None) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    mod = types.ModuleType(module_name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[module_name] = mod
    parts = module_name.rsplit(".", 1)
    if len(parts) == 2:
        parent = sys.modules.get(parts[0])
        if parent is not None:
            setattr(parent, parts[1], mod)
    return mod


def _inject_te_stubs() -> None:
    r"""Create transformer_engine stub with real RoPE + attention fallbacks."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=False, cu_seqlens=None):
        if tensor_format == "bshd" and freqs.ndim == 4 and freqs.shape[1] == 1:
            freqs = freqs.transpose(0, 1)
        cos_ = freqs.cos().to(t.dtype)
        sin_ = freqs.sin().to(t.dtype)
        rot_dim = cos_.shape[-1]
        half = rot_dim // 2
        t_rot = t[..., :rot_dim]
        t_pass = t[..., rot_dim:]
        x1, x2 = t_rot[..., :half], t_rot[..., half:]
        cos_h = cos_[..., :half]
        sin_h = sin_[..., :half]
        o1 = x1 * cos_h - x2 * sin_h
        o2 = x2 * cos_h + x1 * sin_h
        return torch.cat([o1, o2, t_pass], dim=-1)

    class DotProductAttention(nn.Module):
        def __init__(self, num_attention_heads=1, kv_channels=64, attention_dropout=0.0, **kwargs):
            super().__init__()
            self.num_heads = num_attention_heads
            self.head_dim = kv_channels
            self.dropout = attention_dropout

        def forward(self, q, k, v, attn_mask_type="no_mask", **kwargs):
            try:
                from flash_attn import flash_attn_func

                if q.ndim == 4:
                    out = flash_attn_func(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        causal=attn_mask_type == "causal",
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=attn_mask_type == "causal",
                    )
            except Exception:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=attn_mask_type == "causal")
            return out

    class _RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (x.float() * norm).to(x.dtype) * self.weight

        def reset_parameters(self):
            nn.init.ones_(self.weight)

    _inject_stub("transformer_engine", {"__version__": "2.8.0"})
    _inject_stub("transformer_engine.pytorch", {"RMSNorm": _RMSNorm})
    _inject_stub(
        "transformer_engine.pytorch.attention",
        {"DotProductAttention": DotProductAttention, "apply_rotary_pos_emb": apply_rotary_pos_emb},
    )
    _inject_stub(
        "transformer_engine.pytorch.attention.rope",
        {"apply_rotary_pos_emb": apply_rotary_pos_emb},
    )


def _inject_training_stubs() -> None:
    if "megatron" in sys.modules:
        return

    _inject_stub("megatron")
    _inject_stub("megatron.core")
    _inject_stub(
        "megatron.core.parallel_state",
        {
            "get_data_parallel_world_size": lambda: 1,
            "get_data_parallel_rank": lambda: 0,
            "get_tensor_model_parallel_world_size": lambda: 1,
            "get_tensor_model_parallel_rank": lambda: 0,
            "get_context_parallel_world_size": lambda: 1,
            "get_context_parallel_rank": lambda: 0,
            "get_context_parallel_group": lambda: None,
            "is_initialized": lambda: False,
            "model_parallel_is_initialized": lambda: False,
            "get_global_memory_buffer": lambda: None,
            "initialize_model_parallel": lambda *a, **kw: None,
            "destroy_model_parallel": lambda: None,
        },
    )

    if "transformer_engine" not in sys.modules:
        _inject_te_stubs()
    if "transformer_engine_torch" not in sys.modules:
        _inject_stub("transformer_engine_torch")

    if "pytorch3d" not in sys.modules:
        import torch

        _inject_stub("pytorch3d")
        _inject_stub(
            "pytorch3d.transforms",
            {
                "matrix_to_rotation_6d": lambda m: m[..., :2].reshape(*m.shape[:-2], 6),
                "rotation_6d_to_matrix": lambda r: torch.eye(3).expand(*r.shape[:-1], 3, 3),
                "axis_angle_to_matrix": lambda a: torch.eye(3).expand(*a.shape[:-1], 3, 3),
                "matrix_to_axis_angle": lambda m: torch.zeros(*m.shape[:-2], 3),
            },
        )

    log.debug("Injected training-only dependency stubs")


def _find_cosmos_predict2() -> str | None:
    project_root = Path(__file__).resolve().parents[4]
    search_paths = [
        os.environ.get("COSMOS_PREDICT2_PATH", ""),
        str(project_root / "template-projs-toberemoved" / "DreamDojo-main"),
        str(Path.home() / "DreamDojo"),
        str(Path.home() / "cosmos-predict2"),
    ]
    for p in search_paths:
        if p and (Path(p) / "cosmos_predict2" / "__init__.py").exists():
            return p
    return None


def _try_clone_cosmos_predict2() -> str | None:
    import shutil
    import subprocess

    if shutil.which("git") is None:
        return None
    dest = Path.home() / "DreamDojo"
    if dest.exists():
        return None
    log.info("Cloning github.com/NVIDIA/DreamDojo to %s ...", dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/NVIDIA/DreamDojo.git", str(dest)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        if (dest / "cosmos_predict2" / "__init__.py").exists():
            return str(dest)
    except Exception as exc:
        log.warning("Auto-clone failed: %s", exc)
    return None


def ensure_cosmos_predict2() -> None:
    r"""Set up environment so ``import cosmos_predict2`` works for inference."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    os.environ.setdefault("COSMOS_INTERNAL", "1")

    _inject_training_stubs()

    try:
        importlib.import_module("cosmos_predict2")
        log.debug("cosmos_predict2 already importable")
        return
    except (ImportError, RuntimeError):
        pass

    repo_root = _find_cosmos_predict2()
    if repo_root is None:
        repo_root = _try_clone_cosmos_predict2()
    if repo_root is None:
        raise ImportError(
            "cosmos_predict2 not found. Fix with ONE of:\n"
            "  1. git clone https://github.com/NVIDIA/DreamDojo.git ~/DreamDojo\n"
            "  2. export COSMOS_PREDICT2_PATH=/path/to/DreamDojo"
        )

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        log.info("Added %s to sys.path for cosmos_predict2", repo_root)

    cosmos_cuda_spec = importlib.util.find_spec("cosmos_cuda")
    if cosmos_cuda_spec is None:
        _inject_stub("cosmos_cuda", {"__version__": "1.4.1"})

    importlib.import_module("cosmos_predict2")
    log.info("cosmos_predict2 loaded from %s", repo_root)
