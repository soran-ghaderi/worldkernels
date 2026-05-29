r"""Provision python deps (uv) and native packages (git clone) with progress."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from worldkernels.bootstrap import cache
from worldkernels.bootstrap.errors import FetchDisabledError

if TYPE_CHECKING:
    from worldkernels.bootstrap.progress import ProgressController
    from worldkernels.worlds.hub import GitPackage, ModelCard

log = logging.getLogger(__name__)


_EXTRA_SENTINELS: dict[str, str] = {
    "cosmos": "transformers",
    "diffusion": "diffusers",
}


def provision_python_deps(
    card: "ModelCard",
    progress: "ProgressController | None" = None,
    allow_fetch: bool = True,
    target_python: str | None = None,
) -> None:
    extra = card.pip_extra
    if extra is None:
        if progress is not None:
            progress.event("deps", "skipped", "none required")
        return

    sentinel = _EXTRA_SENTINELS.get(extra, extra)
    if target_python is None:
        try:
            importlib.import_module(sentinel)
            if progress is not None:
                progress.event("deps", "skipped", f"{extra} already installed")
            return
        except ImportError:
            pass

    if not allow_fetch or os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise FetchDisabledError(
            f"missing python deps for '{card.adapter}' (extra: {extra})",
            f"install with: pip install 'worldkernels[{extra}]'",
        )

    pkg = f"worldkernels[{extra}]"
    if progress is not None:
        progress.event("deps", "running", f"installing '{pkg}' …")
    log.info("Auto-installing missing dependencies: %s", pkg)

    install_packages([pkg], target_python=target_python, progress=progress)

    if progress is not None:
        progress.event("deps", "done", f"installed {extra}")


def install_packages(
    packages: list[str],
    target_python: str | None = None,
    progress: "ProgressController | None" = None,
    constraints: str | None = None,
) -> None:
    r"""Install pip packages into the target python via uv (pip fallback).

    Args:
        packages: PEP 508 package specs.
        target_python: Python interpreter to install into (defaults to the current one).
        progress: Optional progress controller for live status updates.
        constraints: Optional path to a requirements/lock constraints file.
    """
    if not packages:
        return
    python_exe = target_python or sys.executable
    runner = _select_installer()
    cmd = _build_install_cmd(runner, packages, python_exe, constraints)
    quiet = progress is not None and progress.mode in ("tty", "quiet", "json", "sink")

    log.debug("Running install: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=quiet, text=True)
    if proc.returncode != 0:
        if progress is not None:
            progress.event("deps", "failed", f"install failed (exit {proc.returncode})")
        if quiet:
            sys.stderr.write((proc.stdout or "") + "\n" + (proc.stderr or "") + "\n")
        raise RuntimeError(f"install failed for {packages!r} via {runner} (exit {proc.returncode})")


def _select_installer() -> str:
    if shutil.which("uv") is not None:
        return "uv"
    return "pip"


def _build_install_cmd(
    runner: str, packages: list[str], python_exe: str, constraints: str | None
) -> list[str]:
    if runner == "uv":
        cmd = ["uv", "pip", "install", "--python", python_exe]
        if constraints:
            cmd += ["--constraint", constraints]
        cmd += packages
        return cmd
    cmd = [python_exe, "-m", "pip", "install", "--disable-pip-version-check"]
    if constraints:
        cmd += ["--constraint", constraints]
    cmd += packages
    return cmd


def provision_git_packages(
    card: "ModelCard",
    progress: "ProgressController | None" = None,
    allow_fetch: bool = True,
) -> None:
    if not card.git_packages:
        if progress is not None:
            progress.event("packages", "skipped", "none required")
        return

    if progress is not None:
        progress.event("packages", "running", "checking …")

    installed: list[str] = []
    for gp in card.git_packages:
        path = _ensure_git_package(gp, progress=progress, allow_fetch=allow_fetch)
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        installed.append(gp.name)

    if progress is not None:
        progress.event("packages", "done", ", ".join(installed))


def _ensure_git_package(
    gp: "GitPackage",
    progress: "ProgressController | None" = None,
    allow_fetch: bool = True,
) -> Path | None:
    if gp.env_path_var:
        env_path = os.environ.get(gp.env_path_var)
        if env_path and _is_valid(Path(env_path), gp.import_check):
            return Path(env_path)

    for candidate in _candidate_paths(gp):
        if _is_valid(candidate, gp.import_check):
            return candidate

    dest = cache.packages_dir() / gp.name

    if dest.exists() and _is_valid(dest, gp.import_check):
        return dest

    if not allow_fetch or os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise FetchDisabledError(
            f"missing git package '{gp.name}'",
            f"clone {gp.url} to {dest} or set ${gp.env_path_var}",
        )

    if shutil.which("git") is None:
        raise RuntimeError(
            f"git not found on PATH; cannot fetch {gp.name}. Install git or set ${gp.env_path_var}."
        )

    if progress is not None:
        progress.event("packages", "running", f"cloning {gp.url} …")
    log.info("Cloning %s to %s", gp.url, dest)

    args = ["git", "clone", "--depth=1"]
    if gp.ref:
        args += ["--branch", gp.ref]
    args += [gp.url, str(dest)]

    proc = subprocess.run(args, check=False, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        if progress is not None:
            progress.event("packages", "failed", f"clone failed: {gp.url}")
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"git clone failed for {gp.url}")

    return dest


def _candidate_paths(gp: "GitPackage") -> list[Path]:
    return [
        Path.home() / gp.name,
        Path.home() / gp.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git"),
    ]


def _is_valid(path: Path, import_check: str | None) -> bool:
    if not path.exists():
        return False
    if import_check is None:
        return True
    return (path / import_check.replace(".", "/") / "__init__.py").exists()


@dataclass
class GitPackage:
    r"""Spec for a github-only python package fetched on first use."""

    name: str
    url: str
    import_check: str | None = None
    env_path_var: str | None = None
    ref: str | None = None
