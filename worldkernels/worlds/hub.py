r"""Model hub registry mapping HF repo IDs and short aliases to adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCard:
    r"""Metadata for a known world model variant."""

    adapter: str
    hf_repo: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""


_HUB: dict[str, ModelCard] = {}


def register_model(name: str, card: ModelCard) -> None:
    _HUB[name] = card


def get_model_card(name: str) -> ModelCard | None:
    return _HUB.get(name)


def list_models() -> dict[str, ModelCard]:
    return dict(_HUB)


def resolve_model(
    model_id: str,
    **user_kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    r"""Resolve a model identifier to (adapter_name, merged_kwargs).

    Lookup order:
    1. Exact match in hub (short name or HF repo ID).
    2. Return model_id as-is (fall through to worlds registry).

    User kwargs override hub defaults.
    """
    card = _HUB.get(model_id)
    if card is not None:
        merged = {**card.default_kwargs, **user_kwargs}
        return card.adapter, merged
    return model_id, user_kwargs


def _register_builtins() -> None:
    register_model("dummy", ModelCard(adapter="dummy", description="CPU-safe dummy world for testing"))

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
                hf_repo="nvidia/DreamDojo",
                default_kwargs={"variant": variant},
                description=desc,
            ),
        )

    register_model(
        "nvidia/DreamDojo",
        ModelCard(
            adapter="dreamdojo",
            hf_repo="nvidia/DreamDojo",
            default_kwargs={"variant": "2b_pretrain"},
            description="DreamDojo action-conditioned video world model (default: 2B pretrain)",
        ),
    )

    register_model(
        "dreamdojo",
        ModelCard(
            adapter="dreamdojo",
            hf_repo="nvidia/DreamDojo",
            default_kwargs={"variant": "2b_pretrain"},
            description="DreamDojo action-conditioned video world model (default: 2B pretrain)",
        ),
    )

    register_model(
        "cosmos-predict2",
        ModelCard(
            adapter="cosmos_predict2",
            hf_repo="nvidia/Cosmos-Predict2.5-2B",
            description="Cosmos-Predict2.5-2B text-conditioned video-to-world",
        ),
    )

    register_model(
        "nvidia/Cosmos-Predict2.5-2B",
        ModelCard(
            adapter="cosmos_predict2",
            hf_repo="nvidia/Cosmos-Predict2.5-2B",
            description="Cosmos-Predict2.5-2B text-conditioned video-to-world",
        ),
    )

    register_model(
        "cosmos_predict2",
        ModelCard(
            adapter="cosmos_predict2",
            hf_repo="nvidia/Cosmos-Predict2.5-2B",
            description="Cosmos-Predict2.5-2B text-conditioned video-to-world",
        ),
    )


_register_builtins()
