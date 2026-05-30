r"""Pull / models / rm command implementations."""

from __future__ import annotations

import os


def run_pull(
    model: str,
    variant: str | None = None,
    ckpt_path: str | None = None,
    quiet: bool = False,
) -> None:
    from worldkernels.bootstrap import ProgressController, prepare

    if quiet:
        os.environ["WORLDKERNELS_QUIET"] = "1"

    with ProgressController(mode="quiet" if quiet else "plain") as progress:
        prepared = prepare(model, variant=variant, ckpt_path=ckpt_path, progress=progress)
        progress.finalize(success=True, summary=f"cached {prepared.alias}")


def run_models(show_all: bool = False) -> None:
    from worldkernels.bootstrap import cache

    if show_all:
        from worldkernels.worlds.hub import list_models

        hub = list_models()
        print("Available models (hub):")
        for name, card in sorted(hub.items()):
            desc = f"  {card.description}" if card.description else ""
            print(f"  {name:32s}{desc}")
        return

    manifests = cache.list_manifests()
    if not manifests:
        print(
            "No models cached locally. "
            "Use `worldkernels pull <model>` or `worldkernels models --all`."
        )
        return

    print(f"Local models (cached under {cache.home()}):")
    for m in sorted(manifests, key=lambda x: x.model_id):
        variant_str = f":{m.variant}" if m.variant else ""
        size_str = ""
        if m.hf_repo:
            sz = cache.hf_cache_size_bytes(m.hf_repo)
            if sz:
                size_str = f"  {_human(sz)}"
        print(f"  {m.model_id}{variant_str:20s}  adapter={m.adapter}{size_str}")


def run_rm(model: str, variant: str | None = None) -> None:
    from huggingface_hub import scan_cache_dir

    from worldkernels.bootstrap import cache
    from worldkernels.runtime import envs
    from worldkernels.worlds.hub import get_model_card

    removed_manifest = cache.remove_manifest(model, variant)
    removed_env = envs.remove_env(model)

    card = get_model_card(model)
    hf_repo = card.hf_repo if card and card.hf_repo else (model if "/" in model else None)

    if hf_repo:
        try:
            info = scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == hf_repo:
                    for rev in repo.revisions:
                        info.delete_revisions(rev.commit_hash).execute()
                        print(f"removed {hf_repo} ({rev.commit_hash[:12]})")
        except Exception as exc:
            print(f"warning: could not scan HF cache: {exc}")

    if removed_manifest:
        print(f"removed manifest: {model}{':' + variant if variant else ''}")
    if removed_env:
        print(f"removed isolated env: {model}")
    if not (removed_manifest or removed_env or hf_repo):
        print(f"nothing to remove for '{model}'")


def _human(b: int) -> str:
    size = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"
