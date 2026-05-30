r"""Cache layout: WORLDKERNELS_HOME with packages/, manifests/."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def home() -> Path:
    env = os.environ.get("WORLDKERNELS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "worldkernels"


def packages_dir() -> Path:
    p = home() / "packages"
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifests_dir() -> Path:
    p = home() / "manifests"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Manifest:
    model_id: str
    variant: str | None = None
    adapter: str = ""
    hf_repo: str | None = None
    ckpt_path: str | None = None
    pip_extra: str | None = None
    git_packages: list[str] = field(default_factory=list)
    created_at: str = ""

    def write(self) -> None:
        path = manifests_dir() / f"{_slug(self.model_id, self.variant)}.json"
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read(cls, model_id: str, variant: str | None = None) -> "Manifest | None":
        path = manifests_dir() / f"{_slug(model_id, variant)}.json"
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))


def list_manifests() -> list[Manifest]:
    out: list[Manifest] = []
    for p in manifests_dir().glob("*.json"):
        try:
            out.append(Manifest(**json.loads(p.read_text())))
        except Exception:
            continue
    return out


def remove_manifest(model_id: str, variant: str | None = None) -> bool:
    path = manifests_dir() / f"{_slug(model_id, variant)}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _slug(model_id: str, variant: str | None) -> str:
    s = model_id.replace("/", "_").replace(":", "_")
    if variant:
        s = f"{s}__{variant}"
    return s


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def hf_cache_size_bytes(repo_id: str) -> int:
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        return sum(repo.size_on_disk for repo in info.repos if repo.repo_id == repo_id)
    except Exception:
        return 0
