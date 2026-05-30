r"""Model hub: maps HF repo IDs and short aliases to world models and generators."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

_WAN_NEGATIVE_PROMPT = (
    "overexposed, static, blurred details, subtitles, worst quality, low quality, "
    "JPEG compression residue, ugly, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, cluttered background, three legs, walking backwards"
)


@dataclass(frozen=True)
class GitPackage:
    r"""Spec for a github-only python package fetched on first use."""

    name: str
    url: str
    import_check: str | None = None
    env_path_var: str | None = None
    ref: str | None = None


@dataclass(frozen=True)
class Component:
    r"""Sub-model component with its own pip extra and import sentinel.

    Args:
        name: Component identifier (e.g. ``"wan-vae"``).
        extra: The worldkernels extra that provides it.
        sentinel: A module path used to detect whether the extra is installed.
        deps: PEP 508 specs for the component's deps (used by the resolver).
    """

    name: str
    extra: str
    sentinel: str
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelCard:
    r"""Metadata for a known model.

    Args:
        adapter: World-registry key (``"dreamdojo"``, ``"generator_world"`` …).
        kind: ``"world"`` or ``"generator"``.
        generator: Pipeline-registry key (for kind=generator).
        hf_repo: HuggingFace repo ID.
        default_kwargs: Constructor kwargs merged under user kwargs.
        description: Human-readable summary.
        pip_extra: Optional worldkernels extra to auto-install.
        git_packages: GitHub-only python deps fetched on first use.
        variants: Named variant kwargs (``{"2b_gr1": {"variant": "2b_gr1"}}``).
        auth_required: Whether the HF repo is gated.
        allow_patterns: HF snapshot_download patterns.
        variant_pattern: Per-variant HF download patterns (``{variant}`` is substituted).
    """

    adapter: str
    kind: Literal["world", "generator"] = "world"
    generator: str | None = None
    hf_repo: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    pip_extra: str | None = None
    git_packages: list[GitPackage] = field(default_factory=list)
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)
    auth_required: bool = False
    allow_patterns: list[str] | None = None
    variant_pattern: list[str] | None = None
    isolation: Literal["auto", "shared", "isolated"] = "auto"
    constraints: list[str] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)


_HUB: dict[str, ModelCard] = {}


def register_model(name: str, card: ModelCard) -> None:
    _HUB[name] = card


def get_model_card(name: str) -> ModelCard | None:
    return _HUB.get(name)


def list_models() -> dict[str, ModelCard]:
    return dict(_HUB)


def infer_card_from_hf(repo_id: str) -> ModelCard | None:
    r"""Synthesize a minimal card for an unknown ``org/repo`` reference.

    Probes the HF Hub for the model card metadata; if the repo doesn't exist,
    returns ``None``. Adapter selection is best-effort based on tags and falls
    back to ``"generator_world"`` for diffusion-style repos.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        return None

    try:
        info = HfApi().model_info(repo_id)
    except RepositoryNotFoundError:
        return None
    except Exception as exc:
        log.debug("HF probe failed for %s: %s", repo_id, exc)
        return None

    tags = set(info.tags or [])
    adapter = "generator_world"
    pip_extra = "diffusion"
    if "diffusers" in tags:
        adapter = "generator_world"
        pip_extra = "diffusion"

    return ModelCard(
        adapter=adapter,
        hf_repo=repo_id,
        description=f"auto-inferred from {repo_id}",
        pip_extra=pip_extra,
    )


_EXTRA_SENTINELS: dict[str, str] = {
    "cosmos": "transformers",
    "diffusion": "diffusers",
}


def ensure_model_deps(model_id: str) -> None:
    r"""Auto-install missing pip extras for a model.

    Back-compat shim retained for ``model:inspect`` and tests. New code should
    use ``worldkernels.bootstrap.prepare``, which integrates with the unified
    progress UI.
    """
    import importlib as _importlib

    card = _HUB.get(model_id)
    if card is None or card.pip_extra is None:
        return
    sentinel = _EXTRA_SENTINELS.get(card.pip_extra)
    if sentinel is None:
        return
    try:
        _importlib.import_module(sentinel)
        return
    except ImportError:
        pass

    import os as _os

    if _os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise ImportError(
            f"Missing dependencies for '{model_id}'. "
            f"Install with: pip install 'worldkernels[{card.pip_extra}]'"
        )
    import subprocess as _subprocess
    import sys as _sys

    extra = f"worldkernels[{card.pip_extra}]"
    log.info("Auto-installing missing dependencies: pip install '%s' ...", extra)
    _subprocess.check_call([_sys.executable, "-m", "pip", "install", extra])
    log.info("Dependencies installed successfully.")


def resolve_model(model_id: str, **user_kwargs: Any) -> tuple[str, dict[str, Any]]:
    r"""Resolve a model identifier to ``(world_registry_key, merged_kwargs)``.

    Deprecated; use ``worldkernels.bootstrap.prepare`` for new code. Retained for
    legacy callers that don't need the bootstrap pipeline (e.g., ``model:inspect``).
    """
    card = _HUB.get(model_id)
    if card is None:
        return model_id, user_kwargs
    merged = {**card.default_kwargs, **user_kwargs}
    if card.generator is not None:
        merged.setdefault("generator", card.generator)
    return card.adapter, merged


_DREAMDOJO_GIT = GitPackage(
    name="DreamDojo",
    url="https://github.com/NVIDIA/DreamDojo.git",
    import_check="cosmos_predict2",
    env_path_var="COSMOS_PREDICT2_PATH",
)


def _register_builtins() -> None:
    register_model(
        "dummy",
        ModelCard(adapter="dummy", description="CPU-safe dummy world for testing"),
    )

    _dreamdojo_variants = {
        "2b_pretrain": "DreamDojo 2B pretrained (general)",
        "2b_gr1": "DreamDojo 2B fine-tuned on GR-1 robot",
        "2b_agibot": "DreamDojo 2B fine-tuned on AgiBot",
        "2b_g1": "DreamDojo 2B fine-tuned on G1 robot",
        "2b_yam": "DreamDojo 2B fine-tuned on YAM robot",
        "14b_pretrain": "DreamDojo 14B pretrained (general)",
        "14b_gr1": "DreamDojo 14B fine-tuned on GR-1 robot",
    }

    dreamdojo_card = ModelCard(
        adapter="dreamdojo",
        kind="world",
        hf_repo="nvidia/DreamDojo",
        default_kwargs={"variant": "2b_pretrain"},
        description="DreamDojo action-conditioned video world model",
        pip_extra="cosmos",
        git_packages=[_DREAMDOJO_GIT],
        variants={k: {"variant": k} for k in _dreamdojo_variants},
    )
    register_model("dreamdojo", dreamdojo_card)
    register_model("nvidia/DreamDojo", dreamdojo_card)
    register_model("DreamDojo", dreamdojo_card)

    for variant, desc in _dreamdojo_variants.items():
        short = f"dreamdojo-{variant.replace('_', '-')}"
        register_model(
            short,
            ModelCard(
                adapter="dreamdojo",
                kind="world",
                hf_repo="nvidia/DreamDojo",
                default_kwargs={"variant": variant},
                description=desc,
                pip_extra="cosmos",
                git_packages=[_DREAMDOJO_GIT],
            ),
        )

    cosmos_card = ModelCard(
        adapter="generator_world",
        kind="generator",
        generator="cosmos_predict2",
        hf_repo="nvidia/Cosmos-Predict2.5-2B",
        default_kwargs={"num_inference_steps": 35, "guidance_scale": 7.0},
        description="Cosmos-Predict2.5-2B text-conditioned video generator",
        pip_extra="cosmos",
        git_packages=[_DREAMDOJO_GIT],
    )
    for name in ("cosmos-predict2", "cosmos_predict2", "nvidia/Cosmos-Predict2.5-2B"):
        register_model(name, cosmos_card)

    _wan_variants = {
        "wan2.2-ti2v-5b": (
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            5.0,
            "Wan2.2 TI2V-5B text/image-to-video (dense, 720p)",
        ),
        "wan2.1-i2v-14b": (
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            3.0,
            "Wan2.1 I2V-14B image-to-video (480p)",
        ),
    }
    for short, (repo, flow_shift, desc) in _wan_variants.items():
        card = ModelCard(
            adapter="generator_world",
            kind="generator",
            generator="wan_i2v",
            hf_repo=repo,
            default_kwargs={
                "repo": repo,
                "pipeline_class": "WanImageToVideoPipeline",
                "flow_shift": flow_shift,
                "negative_prompt": _WAN_NEGATIVE_PROMPT,
            },
            description=desc,
            pip_extra="diffusion",
        )
        register_model(short, card)
        register_model(repo, card)


_register_builtins()
