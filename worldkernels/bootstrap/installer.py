r"""Installer dispatch: Shared vs Isolated tier (ADR-012).

The shared installer puts deps into the running interpreter's env (the same
path the lazy bootstrap has always used). The isolated installer materializes
a per-model uv venv and returns its path; the engine then spawns a worker
inside that venv via `RemoteWorld`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from worldkernels.bootstrap.progress import ProgressController
    from worldkernels.worlds.hub import ModelCard

log = logging.getLogger(__name__)

__all__ = ["Installer", "SharedInstaller", "IsolatedInstaller", "InstallResult"]


@dataclass
class InstallResult:
    r"""Where the model's deps now live."""

    tier: str  # "shared" | "isolated"
    env_path: Path | None = None


class Installer(Protocol):
    def install(
        self,
        card: "ModelCard",
        progress: "ProgressController | None",
        allow_fetch: bool,
    ) -> InstallResult: ...


class SharedInstaller:
    r"""Install card's pip_extra (and git packages) into the current interpreter."""

    def install(
        self,
        card: "ModelCard",
        progress: "ProgressController | None",
        allow_fetch: bool,
    ) -> InstallResult:
        from worldkernels.bootstrap import deps

        deps.provision_python_deps(card, progress=progress, allow_fetch=allow_fetch)
        deps.provision_git_packages(card, progress=progress, allow_fetch=allow_fetch)
        return InstallResult(tier="shared", env_path=None)


class IsolatedInstaller:
    r"""Materialize a per-model uv venv and install the card's deps into it."""

    def __init__(self, model_id: str, requirements: list[str], device: str = "cuda") -> None:
        self.model_id = model_id
        self.requirements = requirements
        self.device = device

    def install(
        self,
        card: "ModelCard",
        progress: "ProgressController | None",
        allow_fetch: bool,
    ) -> InstallResult:
        from worldkernels.runtime import envs

        if progress is not None:
            progress.event("isolating", "running", f"materializing env for {self.model_id} …")
        env_path = envs.materialize_env(
            self.model_id,
            self.requirements,
            device=self.device,
            progress=progress,
            allow_fetch=allow_fetch,
        )
        return InstallResult(tier="isolated", env_path=env_path)
