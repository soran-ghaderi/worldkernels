r"""Resolve a user-facing model reference to (ModelCard, variant, ckpt_path)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from worldkernels.bootstrap.errors import ModelNotFoundError

if TYPE_CHECKING:
    from worldkernels.worlds.hub import ModelCard

log = logging.getLogger(__name__)


@dataclass
class ResolvedRef:
    card: "ModelCard"
    name: str
    variant: str | None = None
    ckpt_path: str | None = None


def resolve(ref: str, variant: str | None = None, ckpt_path: str | None = None) -> ResolvedRef:
    r"""Resolve a CLI/HTTP model reference.

    Resolution order:
    1. Local file/dir path (if it exists on disk).
    2. Literal hub key.
    3. Variant-suffix split (``name:variant`` or ``name@variant``).
    4. HF URL strip → repo ID lookup.
    5. Bare HF repo ID (``org/repo``) — looked up in hub, else a synthetic card.
    """
    from worldkernels.worlds.hub import get_model_card, infer_card_from_hf

    if ckpt_path is None and _looks_like_path(ref):
        return _local_ref(ref, variant)

    url_stripped = _strip_hf_url(ref)
    if url_stripped is not None:
        ref = url_stripped

    card = get_model_card(ref)
    if card is not None:
        return _from_card(card, ref, variant, ckpt_path)

    for sep in (":", "@"):
        if sep in ref:
            base, _, suf = ref.rpartition(sep)
            base_card = get_model_card(base)
            if base_card is not None:
                return _from_card(base_card, base, variant or suf, ckpt_path)

    if "/" in ref:
        card = infer_card_from_hf(ref)
        if card is not None:
            return _from_card(card, ref, variant, ckpt_path)

    if _registered_in_worlds(ref):
        from worldkernels.worlds.hub import ModelCard

        synthetic = ModelCard(adapter=ref, description=f"registry-only adapter: {ref}")
        return _from_card(synthetic, ref, variant, ckpt_path)

    raise ModelNotFoundError(
        f"could not resolve model '{ref}'. "
        f"Try a registered alias (`worldkernels models`) or an HF repo id (org/name)."
    )


def _registered_in_worlds(name: str) -> bool:
    try:
        from worldkernels.worlds.registry import _REGISTRY

        return name in _REGISTRY
    except Exception:
        return False


def _from_card(
    card: "ModelCard",
    name: str,
    variant: str | None,
    ckpt_path: str | None,
) -> ResolvedRef:
    if variant is None and card.default_kwargs.get("variant"):
        variant = card.default_kwargs["variant"]
    return ResolvedRef(card=card, name=name, variant=variant, ckpt_path=ckpt_path)


def _local_ref(ref: str, variant: str | None) -> ResolvedRef:
    from worldkernels.worlds.hub import ModelCard

    path = Path(ref).expanduser().resolve()
    suf = path.suffix.lower()
    if suf in {".pt", ".bin", ".safetensors"} or (path.is_dir() and (path / "config.py").exists()):
        adapter = "dreamdojo"
    else:
        adapter = "dreamdojo"
    card = ModelCard(adapter=adapter, description=f"local checkpoint: {path}")
    return ResolvedRef(card=card, name=path.name, variant=variant, ckpt_path=str(path))


def _looks_like_path(ref: str) -> bool:
    if ref.startswith(("./", "../", "/", "~")):
        return True
    p = Path(ref).expanduser()
    return p.exists()


def _strip_hf_url(ref: str) -> str | None:
    prefixes = ("https://huggingface.co/", "http://huggingface.co/", "hf://")
    for pre in prefixes:
        if ref.startswith(pre):
            tail = ref[len(pre) :].split("?")[0].split("#")[0].rstrip("/")
            return tail
    return None
