r"""HuggingFace download tuning.

By default we enable ``hf_transfer`` (Rust-based parallel downloader, 10–50×
faster than single-connection Python downloads). Users on flaky / proxied
networks can opt out with ``HF_HUB_ENABLE_HF_TRANSFER=0`` set before launch.
"""

from __future__ import annotations

import os


def enable_fast_hf_transfer() -> None:
    r"""Set ``HF_HUB_ENABLE_HF_TRANSFER=1`` unless the user explicitly chose otherwise.

    Idempotent. Safe to call from any HF download site.
    """
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
