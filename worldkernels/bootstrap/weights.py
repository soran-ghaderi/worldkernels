r"""Checkpoint provisioning: HF snapshot_download or local path passthrough."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from worldkernels.bootstrap.errors import AuthRequiredError, FetchDisabledError

if TYPE_CHECKING:
    from worldkernels.bootstrap.progress import ProgressController
    from worldkernels.worlds.hub import ModelCard

log = logging.getLogger(__name__)


def provision_weights(
    card: "ModelCard",
    variant: str | None = None,
    ckpt_path: str | None = None,
    progress: "ProgressController | None" = None,
    allow_fetch: bool = True,
) -> str | None:
    if ckpt_path is not None:
        p = Path(ckpt_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {p}")
        if progress is not None:
            progress.event("weights", "done", f"local: {p}")
        return str(p)

    if card.hf_repo is None and card.weights_provider is None:
        if progress is not None:
            progress.event("weights", "skipped", "no hf repo declared")
        return None

    source = card.hf_repo or card.weights_provider
    if not allow_fetch or os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise FetchDisabledError(
            f"missing weights for {source}",
            f"run `worldkernels pull {card.hf_repo or ''}`".strip() + " or pass --ckpt-path",
        )

    if progress is not None:
        progress.event("weights", "running", f"{source} · {variant or ''}".strip(" ·"))

    from worldkernels.bootstrap.hf import enable_fast_hf_transfer

    enable_fast_hf_transfer()

    if card.weights_provider is not None:
        local_path = _load_weights_provider(card.weights_provider)(variant)
        if progress is not None:
            progress.event("weights", "done", str(source))
        return local_path

    assert card.hf_repo is not None

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    allow_patterns = card.allow_patterns
    if variant and card.variant_pattern:
        allow_patterns = [p.format(variant=variant) for p in card.variant_pattern]

    try:
        local_dir = snapshot_download(
            card.hf_repo,
            allow_patterns=allow_patterns,
            repo_type="model",
        )
    except GatedRepoError as exc:
        raise AuthRequiredError(card.hf_repo) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"HF repo not found: {card.hf_repo}") from exc

    if progress is not None:
        progress.event("weights", "done", f"{card.hf_repo}")
    return local_dir


def _load_weights_provider(spec: str) -> Callable[[str | None], str]:
    import importlib

    module_path, _, func_name = spec.partition(":")
    return getattr(importlib.import_module(module_path), func_name)
