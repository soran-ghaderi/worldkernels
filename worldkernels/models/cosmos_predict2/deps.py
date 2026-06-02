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
import importlib.abc
import importlib.metadata
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

log = logging.getLogger(__name__)

_setup_done = False

_TRAINING_ONLY = frozenset(
    {
        # telemetry / experiment tracking / system monitoring
        "wandb",
        "pynvml",
        "psutil",
        "git",
        # storage / checkpoint mirrors (S3, NVIDIA multi-storage)
        "boto3",
        "botocore",
        "multistorageclient",
        "gdown",
        # training data loaders / video-IO / augmentation (pulled by groot_dreams via the
        # action-conditioned config, which inference overrides to mock data)
        "webdataset",
        "decord",
        "av",
        "mediapy",
        "albumentations",
        "lerobot",
        "nvidia_dali",
        "trimesh",
        # guardrail / aux models (not on the core diffusion path)
        "retinaface",
        "better_profanity",
        # text pre-processing for the off-by-default text encoders (umt5/reason1); imported by
        # the conditioner but only *called* when text conditioning runs (text_encoder='on')
        "ftfy",
        "nltk",
        # training-loop / plotting / analysis
        "straggler",
        "matplotlib",
        "pandas",
    }
)
# `lightning` / `pytorch_lightning` are handled by a separate FUNCTIONAL stub (below) rather than
# no-op'd: cosmos subclasses ``LightningModule`` for instantiated ``nn.Module`` components.
# Deliberately NOT here (a no-op stub would be WRONG): guarded optional compute kernels
# (natten, flash_attn, flash_attn_3, cudnn, triton — cosmos try/excepts these and falls back,
# so a fake module defeats the guard; flash_attn is handled separately via a metadata shim),
# structural/base-class libs (iopath, pydantic, attrs — must be real), and dual-use IO
# (imageio — also worldkernels' own video output; install it if a run needs it).
# This set covers every module-level third-party import across cosmos_predict2/_src + groot_dreams
# that the [cosmos] extra does not install. Re-audit on cosmos upgrades:
#   grep -rhoE '^(import|from) [a-z]' ~/DreamDojo/{cosmos_predict2/_src,groot_dreams}


class _NoOpMeta(type):
    r"""Metaclass: lets `_NoOp` act as a base class, and stay falsy / attribute-permissive."""

    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_attr(cls, name)

    def __bool__(cls) -> bool:
        return False

    def __iter__(cls):
        return iter(())


class _NoOp(metaclass=_NoOpMeta):
    r"""Universal no-op value: a class so it can be **subclassed** (cosmos does
    ``class X(webdataset.WebLoader)``), yet also callable, attribute-permissive and falsy."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_attr(self, name)

    def __call__(self, *args: object, **kwargs: object):
        return self

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter(())


def _stub_attr(owner: object, name: str) -> type:
    r"""A distinct, cached `_NoOp` subclass per name.

    Distinct identity per name so cosmos's multiple inheritance from stub symbols
    (``class WebDataset(DataPipeline, FluidInterface)``) is well-formed instead of
    ``class WebDataset(_NoOp, _NoOp)``. Keeps `_NoOpMeta`, so it stays falsy /
    callable / iterable / attribute-permissive. Cached on ``owner`` so repeat access
    is identity-stable across ``importlib.reload``.
    """
    stub = _NoOpMeta(name, (_NoOp,), {})
    setattr(owner, name, stub)
    return stub


class _NoOpModule(types.ModuleType):
    r"""Black-hole module: satisfies ``import`` and resolves any symbol to a `_NoOp` subclass.

    For training/telemetry/data libs (wandb, webdataset, …) that cosmos_predict2 hard-imports at
    module level but inference never meaningfully uses. Each symbol resolves to a distinct `_NoOp`
    subclass — a class, so they work both as values (``wandb.log(...)``, falsy ``wandb.run``) and
    as base classes (``class X(webdataset.WebLoader)``); distinct identity per name keeps multiple
    inheritance (``class WebDataset(DataPipeline, FluidInterface)``) well-formed. The module itself
    stays falsy / callable / iterable.
    """

    __path__: list = []

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_attr(self, name)

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class _TrainingStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    r"""Resolve any absent module whose top-level package is training-only to a no-op.

    Appended to ``sys.meta_path`` so the real finders win first: this only fires when a module
    is genuinely not installed, and it covers arbitrary submodules (``botocore.config``,
    ``multistorageclient.types``) without enumerating them.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in _TRAINING_ONLY:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        log.debug("stubbing absent training-only dependency: %s", spec.name)
        return _NoOpModule(spec.name)

    def exec_module(self, module) -> None:
        pass


def _install_training_stub_finder() -> None:
    if not any(isinstance(f, _TrainingStubFinder) for f in sys.meta_path):
        sys.meta_path.append(_TrainingStubFinder())


def _flash_attn_unavailable(*args, **kwargs):
    raise RuntimeError(
        "flash_attn is not installed. The DreamDojo DiT uses torch SDPA and does not need it; "
        "it is stubbed so the (unused) reason1 text encoder can import. Install flash_attn to run "
        "the reason1 text encoder (text_encoder='on')."
    )


class _FlashAttnStubModule(types.ModuleType):
    r"""Importable flash_attn whose every symbol RAISES when called (never returns garbage)."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _flash_attn_unavailable


class _FlashAttnDistribution(importlib.metadata.Distribution):
    r"""Synthetic dist metadata so flash_attn looks installed (version 2.8.0) to transformers."""

    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            return "Metadata-Version: 2.1\nName: flash-attn\nVersion: 2.8.0\n"
        if filename == "top_level.txt":
            return "flash_attn\n"
        return None

    def locate_file(self, path):  # type: ignore[override]
        return path


class _FlashAttnStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    r"""Make flash_attn both importable (symbols raise on call) and a registered distribution.

    Implementing ``find_distributions`` is the key: transformers' whole ``is_flash_attn_*``
    family derives from ``importlib.metadata`` (``packages_distributions``), so one consistent
    fake distribution satisfies all of them without patching any transformers internals.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flash_attn" or fullname.startswith("flash_attn."):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _FlashAttnStubModule(spec.name)

    def exec_module(self, module) -> None:
        pass

    def find_distributions(self, context=None):
        name = getattr(context, "name", None) if context is not None else None
        if name in (None, "flash_attn", "flash-attn"):
            yield _FlashAttnDistribution()


def _install_flash_attn_shim() -> None:
    r"""Let the unused reason1 text encoder import without flash_attn (the DiT uses torch SDPA).

    cosmos_predict2's Qwen2.5-VL encoder hard-requires flash_attn at import (an ``assert`` plus
    ``transformers`` flash-attn checks) even though inference never instantiates it
    (``text_encoder`` defaults off). We make flash_attn look like an installed distribution whose
    symbols raise if ever *called* — so guarded paths fall back to SDPA and nothing silently uses
    a fake kernel — which satisfies transformers' entire detection family via one metadata source
    of truth. No-op when real flash_attn is installed; ``flash_attn_3`` is left absent so the DiT's
    own guarded fast-path still falls back to SDPA.
    """
    if any(isinstance(f, _FlashAttnStubFinder) for f in sys.meta_path):
        return
    if importlib.util.find_spec("flash_attn") is not None:
        return
    sys.meta_path.append(_FlashAttnStubFinder())
    _iu = sys.modules.get("transformers.utils.import_utils")
    _mapping = getattr(_iu, "PACKAGE_DISTRIBUTION_MAPPING", None) if _iu is not None else None
    if isinstance(_mapping, dict):
        _mapping.setdefault("flash_attn", ["flash-attn"])


_LIGHTNING_STUBS: dict[str, type] = {}


def _lightning_attr(name: str) -> type:
    r"""Real ``nn.Module`` base for ``LightningModule``; a cached empty class for anything else.

    Lightning is used *structurally* (base classes for instantiated components), so each symbol
    must be a subclassable class, and ``LightningModule`` specifically an ``nn.Module`` so the
    components register their submodules.
    """
    if name not in _LIGHTNING_STUBS:
        if name == "LightningModule":
            import torch.nn as nn

            class LightningModule(nn.Module):
                def log(self, *args: object, **kwargs: object) -> None:
                    return None

                def log_dict(self, *args: object, **kwargs: object) -> None:
                    return None

                def save_hyperparameters(self, *args: object, **kwargs: object) -> None:
                    return None

                @property
                def global_step(self) -> int:
                    return 0

            _LIGHTNING_STUBS[name] = LightningModule
        else:
            _LIGHTNING_STUBS[name] = type(name, (), {})
    return _LIGHTNING_STUBS[name]


class _LightningStubModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _lightning_attr(name)


class _LightningStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    r"""Functional stub for the Lightning training framework.

    No-op'ing lightning is wrong: cosmos does ``class LAM(LightningModule)`` for components that
    are instantiated as ``nn.Module``s (a module instance is not subclassable). This resolves any
    ``lightning`` / ``pytorch_lightning`` submodule to a stub whose ``LightningModule`` is a real
    ``nn.Module`` and whose other symbols are subclassable empty classes.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in ("lightning", "pytorch_lightning"):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _LightningStubModule(spec.name)

    def exec_module(self, module) -> None:
        pass


def _install_lightning_stub_finder() -> None:
    if any(isinstance(f, _LightningStubFinder) for f in sys.meta_path):
        return
    if importlib.util.find_spec("lightning") is not None:
        return
    sys.meta_path.append(_LightningStubFinder())


def _inject_stub(module_name: str, attrs: dict | None = None) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    mod = types.ModuleType(module_name)
    mod.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
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

    _install_training_stub_finder()
    _install_lightning_stub_finder()

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
    _install_flash_attn_shim()

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
