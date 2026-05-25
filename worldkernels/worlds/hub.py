r"""Model hub: maps HF repo IDs and short aliases to world models and generators."""

from __future__ import annotations

import importlib as _importlib
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
class ModelCard:
    r"""Metadata for a known model.

    Args:
        adapter: World-registry key of the world class. For generators this is
            ``"generator_world"``; the engine wraps the generator pipeline.
        kind: ``"world"`` for a true world model, ``"generator"`` for a one-shot
            video generator surfaced via ``GeneratorWorld``.
        generator: Pipeline-registry key of the wrapped generator (kind=generator).
        hf_repo: HuggingFace repo ID, if any.
        default_kwargs: Constructor kwargs merged under user kwargs.
        description: Human-readable summary.
        pip_extra: Optional ``worldkernels`` extra to auto-install on load.
    """

    adapter: str
    kind: Literal["world", "generator"] = "world"
    generator: str | None = None
    hf_repo: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    pip_extra: str | None = None


_HUB: dict[str, ModelCard] = {}


def register_model(name: str, card: ModelCard) -> None:
    _HUB[name] = card


def get_model_card(name: str) -> ModelCard | None:
    return _HUB.get(name)


def list_models() -> dict[str, ModelCard]:
    return dict(_HUB)


_EXTRA_SENTINELS: dict[str, str] = {
    "cosmos": "transformers",
    "diffusion": "diffusers",
}


def ensure_model_deps(model_id: str) -> None:
    r"""Auto-install missing pip extras required by a model."""
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
    import os

    if os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise ImportError(
            f"Missing dependencies for '{model_id}'. "
            f"Install with: pip install 'worldkernels[{card.pip_extra}]'"
        )
    import subprocess
    import sys

    extra = f"worldkernels[{card.pip_extra}]"
    log.info("Auto-installing missing dependencies: pip install '%s' ...", extra)
    subprocess.check_call([sys.executable, "-m", "pip", "install", extra])
    log.info("Dependencies installed successfully.")


def resolve_model(model_id: str, **user_kwargs: Any) -> tuple[str, dict[str, Any]]:
    r"""Resolve a model identifier to ``(world_registry_key, merged_kwargs)``.

    Hub default kwargs are merged under user kwargs. For a generator card the
    wrapped generator key is injected so the engine constructs a ``GeneratorWorld``.
    """
    card = _HUB.get(model_id)
    if card is None:
        return model_id, user_kwargs
    merged = {**card.default_kwargs, **user_kwargs}
    if card.generator is not None:
        merged.setdefault("generator", card.generator)
    return card.adapter, merged


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
            ),
        )

    _dreamdojo_card = ModelCard(
        adapter="dreamdojo",
        kind="world",
        hf_repo="nvidia/DreamDojo",
        default_kwargs={"variant": "2b_pretrain"},
        description="DreamDojo action-conditioned video world model (default: 2B pretrain)",
        pip_extra="cosmos",
    )
    register_model("dreamdojo", _dreamdojo_card)
    register_model("nvidia/DreamDojo", _dreamdojo_card)

    _cosmos_card = ModelCard(
        adapter="generator_world",
        kind="generator",
        generator="cosmos_predict2",
        hf_repo="nvidia/Cosmos-Predict2.5-2B",
        default_kwargs={"num_inference_steps": 35, "guidance_scale": 7.0},
        description="Cosmos-Predict2.5-2B text-conditioned video generator",
        pip_extra="cosmos",
    )
    for name in ("cosmos-predict2", "cosmos_predict2", "nvidia/Cosmos-Predict2.5-2B"):
        register_model(name, _cosmos_card)

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
