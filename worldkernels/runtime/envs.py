r"""Per-model uv venv materialization under ``WORLDKERNELS_HOME/envs/``."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from worldkernels.bootstrap import cache
from worldkernels.bootstrap.errors import FetchDisabledError
from worldkernels.runtime.locks import EnvLock, deps_hash, torch_abi_tag

if TYPE_CHECKING:
    from worldkernels.bootstrap.progress import ProgressController

log = logging.getLogger(__name__)

__all__ = ["envs_dir", "env_path", "venv_python", "materialize_env", "remove_env"]


def envs_dir() -> Path:
    p = cache.home() / "envs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_path(model_id: str) -> Path:
    return envs_dir() / _slug(model_id)


def venv_python(model_id: str) -> Path:
    base = env_path(model_id)
    if os.name == "nt":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def materialize_env(
    model_id: str,
    requirements: list[str],
    device: str,
    progress: "ProgressController | None" = None,
    allow_fetch: bool = True,
) -> Path:
    r"""Create or reuse an isolated uv venv for ``model_id``.

    Returns the venv root path. Idempotent: if a lock file matches the requested
    requirements + torch ABI + device, returns the existing path without
    touching the venv.
    """
    base = env_path(model_id)
    abi = torch_abi_tag()
    new_hash = deps_hash(requirements, abi, device)

    existing = EnvLock.read(model_id)
    if existing is not None and existing.deps_hash == new_hash and venv_python(model_id).exists():
        if progress is not None:
            progress.event("isolating", "skipped", f"cached env: {base}")
        return base

    if not allow_fetch or os.environ.get("WORLDKERNELS_NO_AUTO_INSTALL"):
        raise FetchDisabledError(
            f"missing isolated env for {model_id!r}",
            f"run `worldkernels pull {model_id}` to materialize it",
        )

    if shutil.which("uv") is None:
        raise RuntimeError(
            "uv not found on PATH; required for per-model env materialization. "
            "uv is a core worldkernels dep — try `pip install --force-reinstall worldkernels`."
        )

    if progress is not None:
        progress.event("isolating", "running", f"creating {base} …")
    log.info("Materializing per-model venv at %s", base)

    if base.exists():
        shutil.rmtree(base)

    _run(["uv", "venv", str(base)], progress=progress)

    if requirements:
        py = str(venv_python(model_id))
        if progress is not None:
            progress.event("isolating", "running", f"installing {len(requirements)} deps …")
        _run(["uv", "pip", "install", "--python", py, *requirements], progress=progress)

    lock = EnvLock(
        model_id=model_id,
        deps_hash=new_hash,
        torch_abi=abi,
        device=device,
        requirements=requirements,
    )
    lock.write()

    if progress is not None:
        progress.event("isolating", "done", f"env ready: {base}")
    return base


def remove_env(model_id: str) -> bool:
    base = env_path(model_id)
    lock_path = EnvLock(model_id=model_id, deps_hash="", torch_abi="", device="").path
    removed = False
    if base.exists():
        shutil.rmtree(base)
        removed = True
    if lock_path.exists():
        lock_path.unlink()
        removed = True
    return removed


def _run(cmd: list[str], progress: "ProgressController | None" = None) -> None:
    quiet = progress is not None and progress.mode in ("tty", "quiet", "json", "sink")
    proc = subprocess.run(cmd, check=False, capture_output=quiet, text=True)
    if proc.returncode != 0:
        if quiet:
            sys.stderr.write((proc.stdout or "") + "\n" + (proc.stderr or "") + "\n")
        raise RuntimeError(f"{' '.join(cmd[:3])} failed (exit {proc.returncode})")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")
