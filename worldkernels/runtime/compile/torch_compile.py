r"""Regional ``torch.compile``.

A video DiT is a stack of identical transformer blocks wrapped by one-off
embedding and projection layers. Compiling only the repeated block — not the
whole model — keeps compile time bounded and avoids recompiles when outer
shapes vary. A module opts in by listing the attribute names of its repeated
``nn.ModuleList``s in ``_repeated_blocks``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["regionally_compile"]


def regionally_compile(
    module: "nn.Module",
    *,
    mode: str = "reduce-overhead",
    repeated_block_attrs: list[str] | None = None,
) -> "nn.Module":
    r"""Compile a module's repeated blocks in place, leaving one-off ops eager.

    Args:
        module: The model to compile.
        mode: ``torch.compile`` mode.
        repeated_block_attrs: Attribute names of repeated ``nn.ModuleList``s.
            Defaults to ``module._repeated_blocks``; if neither is present the
            whole module is compiled.

    Returns:
        The same ``module``, with its repeated blocks replaced by compiled ones.
    """
    import torch

    from worldkernels.config.active import get_active_config

    if not get_active_config().torch_compile:
        return module

    attrs = repeated_block_attrs or getattr(module, "_repeated_blocks", None)
    if not attrs:
        return torch.compile(module, mode=mode)  # type: ignore[return-value]

    for attr in attrs:
        block_list = getattr(module, attr, None)
        if block_list is None:
            continue
        for i in range(len(block_list)):
            block_list[i] = torch.compile(block_list[i], mode=mode)
    return module
