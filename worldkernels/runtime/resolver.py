r"""Dependency resolver: decide shared vs isolated tier (ADR-012).

Wraps ``uv pip compile`` against the union of currently-loaded model constraints
plus the incoming model's constraints. Pure function over constraints — no
side effects on the host environment.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from worldkernels.core.errors import WorldKernelError

if TYPE_CHECKING:
    from worldkernels.worlds.hub import ModelCard

log = logging.getLogger(__name__)

__all__ = [
    "InstallPlan",
    "SharedPlan",
    "IsolatedPlan",
    "IncompatibleDependenciesError",
    "resolve_install_plan",
]


@dataclass(frozen=True)
class SharedPlan:
    r"""Resolver verdict: pack into the shared (current) env. ``deltas`` lists
    the extra packages to install beyond what's already there."""

    deltas: tuple[str, ...] = ()
    resolved_requirements: str = ""


@dataclass(frozen=True)
class IsolatedPlan:
    r"""Resolver verdict: materialize a dedicated venv for this model."""

    reason: str
    conflicting: tuple[str, ...] = field(default_factory=tuple)
    requirements: tuple[str, ...] = field(default_factory=tuple)


InstallPlan = SharedPlan | IsolatedPlan


class IncompatibleDependenciesError(WorldKernelError):
    r"""Raised when ``isolation='shared'`` is forced but the resolver returns IsolatedPlan."""


def resolve_install_plan(
    incoming: "ModelCard",
    loaded: list["ModelCard"],
) -> InstallPlan:
    r"""Decide whether ``incoming`` can share an env with currently-loaded models.

    Args:
        incoming: ModelCard about to be loaded.
        loaded: ModelCards currently loaded in the shared env (excludes isolated ones).
    """
    if incoming.isolation == "isolated":
        return IsolatedPlan(
            reason="card.isolation='isolated' forced",
            requirements=tuple(_card_requirements(incoming)),
        )

    requirements = _union_requirements(loaded + [incoming])
    if not requirements:
        return SharedPlan(deltas=tuple(_card_requirements(incoming)))

    if shutil.which("uv") is None:
        log.warning("uv not on PATH; resolver falls back to optimistic SharedPlan")
        return SharedPlan(deltas=tuple(_card_requirements(incoming)))

    ok, output = _compile_requirements(requirements)
    if ok:
        return SharedPlan(deltas=tuple(_card_requirements(incoming)), resolved_requirements=output)

    if incoming.isolation == "shared":
        raise IncompatibleDependenciesError(
            f"Cannot share env with currently-loaded models. Conflict:\n{output}"
        )

    return IsolatedPlan(
        reason=output.strip().splitlines()[-1] if output.strip() else "unsatisfiable constraints",
        conflicting=tuple(requirements),
        requirements=tuple(_card_requirements(incoming)),
    )


def _card_requirements(card: "ModelCard") -> list[str]:
    reqs: list[str] = list(card.constraints)
    for comp in card.components:
        reqs.extend(comp.deps)
    return reqs


def _union_requirements(cards: list["ModelCard"]) -> list[str]:
    seen: dict[str, None] = {}
    for card in cards:
        for req in _card_requirements(card):
            if req not in seen:
                seen[req] = None
    return list(seen)


def _compile_requirements(requirements: list[str]) -> tuple[bool, str]:
    r"""Run ``uv pip compile`` against ``requirements``; return ``(ok, stdout_or_stderr)``."""
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as f:
        f.write("\n".join(requirements) + "\n")
        in_path = Path(f.name)
    try:
        proc = subprocess.run(
            ["uv", "pip", "compile", "--quiet", str(in_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if proc.returncode == 0:
            return True, proc.stdout
        return False, proc.stderr or proc.stdout
    except subprocess.TimeoutExpired:
        return False, "uv pip compile timed out"
    finally:
        in_path.unlink(missing_ok=True)
