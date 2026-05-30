r"""bootstrap.prepare — single entrypoint that drives all five phases."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from worldkernels.bootstrap import cache, deps, resolve, weights
from worldkernels.bootstrap.progress import ProgressController

log = logging.getLogger(__name__)


@dataclass
class PreparedModel:
    adapter: str
    alias: str
    variant: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ckpt_path: str | None = None


def prepare(
    ref: str,
    variant: str | None = None,
    ckpt_path: str | None = None,
    progress: ProgressController | None = None,
    allow_fetch: bool = True,
    **adapter_kwargs: Any,
) -> PreparedModel:
    r"""Resolve a ref and provision everything needed to load it.

    Drives the five phases in order: resolve → deps → packages → weights → init
    (the ``init`` phase is just marked as ``running`` here; the caller does the
    actual instantiate/warmup so it can update the phase to ``done``).
    """
    owns_progress = progress is None
    if progress is None:
        progress = ProgressController(mode="auto").__enter__()
    assert progress is not None

    try:
        progress.event("resolve", "running", ref)
        resolved = resolve.resolve(ref, variant=variant, ckpt_path=ckpt_path)
        variant_str = f" (variant {resolved.variant})" if resolved.variant else ""
        progress.event("resolve", "done", f"{resolved.name}{variant_str}")

        deps.provision_python_deps(resolved.card, progress=progress, allow_fetch=allow_fetch)
        deps.provision_git_packages(resolved.card, progress=progress, allow_fetch=allow_fetch)

        local_path = weights.provision_weights(
            resolved.card,
            variant=resolved.variant,
            ckpt_path=resolved.ckpt_path,
            progress=progress,
            allow_fetch=allow_fetch,
        )

        progress.event("init", "running", "loading …")

        merged: dict[str, Any] = {**resolved.card.default_kwargs}
        if resolved.variant is not None:
            merged["variant"] = resolved.variant
        if local_path is not None and resolved.ckpt_path is not None:
            merged["ckpt_path"] = local_path
        if resolved.card.generator is not None:
            merged.setdefault("generator", resolved.card.generator)
        merged.update(adapter_kwargs)

        alias = adapter_kwargs.get("alias") or _alias_for(resolved)
        manifest = cache.Manifest(
            model_id=resolved.name,
            variant=resolved.variant,
            adapter=resolved.card.adapter,
            hf_repo=resolved.card.hf_repo,
            ckpt_path=local_path,
            pip_extra=resolved.card.pip_extra,
            git_packages=[gp.name for gp in resolved.card.git_packages],
        )
        manifest.write()

        return PreparedModel(
            adapter=resolved.card.adapter,
            alias=alias,
            variant=resolved.variant,
            kwargs=merged,
            ckpt_path=local_path,
        )
    finally:
        if owns_progress:
            progress.__exit__(None, None, None)


def _alias_for(resolved: resolve.ResolvedRef) -> str:
    base = resolved.name.split("/")[-1]
    if resolved.variant:
        return f"{base}:{resolved.variant}"
    return base
