r"""Classifier-free-guidance parallelism.

A diffusion step under CFG runs the transformer twice — once on the conditional
input, once on the unconditional. With ``cfg_parallel_size == 2`` the two
forwards run on separate ranks and their outputs are gathered, halving the
per-step latency of the guided forward. With size 1 both run locally.
"""

from __future__ import annotations

from typing import Callable

import torch

from worldkernels.distributed.parallel_state import (
    get_cfg_parallel_group,
    get_cfg_parallel_rank,
    get_cfg_parallel_world_size,
)

__all__ = ["CFGParallelMixin"]


class CFGParallelMixin:
    r"""Mixin giving a pipeline a CFG-parallel guided forward.

    The host pipeline calls `cfg_parallel_forward()` with a model callable
    and the conditional / unconditional keyword arguments.
    """

    @staticmethod
    def cfg_parallel_enabled() -> bool:
        return get_cfg_parallel_world_size() == 2

    @staticmethod
    def cfg_parallel_forward(
        model: Callable[..., torch.Tensor],
        cond_kwargs: dict,
        uncond_kwargs: dict,
        guidance_scale: float,
    ) -> torch.Tensor:
        r"""Run the guided forward, splitting cond/uncond across CFG ranks.

        Returns the guided prediction
        \( \epsilon_u + s\,(\epsilon_c - \epsilon_u) \) for guidance scale \(s\).
        """
        if get_cfg_parallel_world_size() == 1:
            cond = model(**cond_kwargs)
            uncond = model(**uncond_kwargs)
            return uncond + guidance_scale * (cond - uncond)

        import torch.distributed as dist

        local = model(**(cond_kwargs if get_cfg_parallel_rank() == 0 else uncond_kwargs))
        gathered = [torch.empty_like(local), torch.empty_like(local)]
        dist.all_gather(gathered, local.contiguous(), group=get_cfg_parallel_group())
        cond, uncond = gathered[0], gathered[1]
        return uncond + guidance_scale * (cond - uncond)
