r"""HuggingFace download tuning.

Prefer the fastest transfer backend that is actually installed: Xet
(``hf_xet`` + ``HF_XET_HIGH_PERFORMANCE``) on modern hubs, else the legacy
``hf_transfer`` Rust downloader. Setting ``HF_HUB_ENABLE_HF_TRANSFER`` is avoided
unless ``hf_transfer`` is present, since recent ``huggingface_hub`` deprecated it
(it warns and ignores the flag when Xet is the active backend).
"""

from __future__ import annotations

import importlib.util as _util
import os


def enable_fast_hf_transfer() -> None:
    r"""Enable the fastest available HF transfer backend. Idempotent.

    Xet's high-performance mode ramps download concurrency aggressively. On
    *unauthenticated* transfers the hub rate-limits and resets connections, which can
    deadlock ``hf_xet`` (sockets stuck in CLOSE-WAIT, no progress). So we gate it on a
    token being present; without one, Xet still runs at its safe default concurrency.
    """
    if _util.find_spec("hf_xet") is not None:
        if _has_hf_token():
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    elif _util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return get_token() is not None
    except Exception:
        return False
