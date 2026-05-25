r"""Flow-matching UniPC multistep scheduler for Wan-family video diffusion.

Wan video models are trained with rectified-flow / flow-matching, so sampling
follows the probability-flow ODE rather than a DDPM reverse chain. This is an
owned specialization of diffusers' UniPC multistep solver configured for that
prediction type, including the resolution-dependent ``flow_shift`` timestep
warp Wan applies. Keeping it as a worldkernels class gives one home for
scheduler defaults and the eventual hand-written solver.
"""

from __future__ import annotations

from diffusers import UniPCMultistepScheduler

__all__ = ["FlowUniPCMultistepScheduler"]


class FlowUniPCMultistepScheduler(UniPCMultistepScheduler):
    r"""UniPC multistep solver in flow-matching mode for Wan video models."""

    @classmethod
    def for_wan(
        cls,
        *,
        flow_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ) -> "FlowUniPCMultistepScheduler":
        r"""Build a scheduler with Wan's flow-matching defaults.

        Args:
            flow_shift: Timestep-schedule shift \(s\). The schedule is warped by
                \(t \mapsto s\,t / (1 + (s - 1)\,t)\); larger \(s\) spends more
                steps at high noise. Wan uses ~3.0 at 480p, ~5.0 at 720p.
            num_train_timesteps: Training timestep count.
        """
        return cls(
            num_train_timesteps=num_train_timesteps,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=flow_shift,
        )
