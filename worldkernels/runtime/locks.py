r"""Per-model lockfile storage under ``WORLDKERNELS_HOME/locks/``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from worldkernels.bootstrap import cache


def locks_dir() -> Path:
    p = cache.home() / "locks"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class EnvLock:
    r"""Lockfile metadata for a per-model isolated venv."""

    model_id: str
    deps_hash: str
    torch_abi: str
    device: str
    requirements: list[str] = field(default_factory=list)
    resolved: str = ""

    @property
    def path(self) -> Path:
        return locks_dir() / f"{_slug(self.model_id)}.lock.json"

    def write(self) -> None:
        self.path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read(cls, model_id: str) -> "EnvLock | None":
        p = locks_dir() / f"{_slug(model_id)}.lock.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))


def deps_hash(deps: list[str], torch_abi: str, device: str) -> str:
    payload = "\n".join(sorted(deps)) + f"|{torch_abi}|{device}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def torch_abi_tag() -> str:
    r"""Return a short tag that changes whenever torch's binary ABI changes."""
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None) or "nocuda"
        hip = getattr(torch.version, "hip", None) or "nohip"
        return f"{torch.__version__}-cu{cuda}-hip{hip}"
    except ImportError:
        return "no-torch"


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")
