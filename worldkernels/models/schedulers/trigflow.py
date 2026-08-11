r"""TrigFlow few-step sampling for Self-Forcing-distilled causal students.

TrigFlow parameterizes the noise level as an angle \(t = \arctan(\sigma/\sigma_d)\)
so a sample decomposes as \(x_t = \cos(t)\,x_0/\sigma_d + \sin(t)\,\epsilon\).
The distilled DreamDojo student denoises each latent frame in a few steps along
a fixed angle ladder, with rectified-flow preconditioning coefficients derived
from the same angle.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "TRIGFLOW_TIMES_4STEP",
    "trigflow_time",
    "trigflow_scaling",
    "trigflow_step",
]

TRIGFLOW_TIMES_4STEP: tuple[float, ...] = (
    math.pi / 2,
    math.atan(15.0),
    math.atan(5.0),
    math.atan(5.0 / 3.0),
)
r"""Student sampling angles, pure noise (\(\pi/2\)) down to \(\arctan(5/3)\)."""


def trigflow_time(sigma: float, sigma_data: float = 1.0) -> float:
    r"""Angle for a noise level: \(t = \arctan(\sigma/\sigma_d)\)."""
    return math.atan(sigma / sigma_data)


def trigflow_scaling(
    trigflow_t: torch.Tensor, sigma_data: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Rectified-flow preconditioning coefficients for a TrigFlow angle.

    Computed in float64 and cast back, matching the reference exactly:
    $$c_\text{skip} = c_\text{in} = \frac{\sigma_d}{\cos t + \sigma_d \sin t},\quad
    c_\text{out} = -\sigma_d \sin t \cdot c_\text{in} / \sigma_d,\quad
    c_\text{noise} = \sigma_d \sin t \cdot c_\text{in} / \sigma_d$$

    The denoiser evaluates \(x_0 = c_\text{skip} x_t + c_\text{out}\,
    F(c_\text{in} x_t, c_\text{noise})\).

    Returns:
        ``(c_skip, c_out, c_in, c_noise)`` in the input dtype.
    """
    dtype = trigflow_t.dtype
    t = trigflow_t.to(torch.float64)
    denom = torch.cos(t) + sigma_data * torch.sin(t)
    c_skip = sigma_data / denom
    c_out = -sigma_data * torch.sin(t) / denom
    c_in = sigma_data / denom
    c_noise = sigma_data * torch.sin(t) / denom
    return c_skip.to(dtype), c_out.to(dtype), c_in.to(dtype), c_noise.to(dtype)


def trigflow_step(
    x0_pred: torch.Tensor,
    noise: torch.Tensor,
    t_next: torch.Tensor,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    r"""Re-noise the current \(x_0\) prediction to the next angle.

    $$x_{t'} = \cos(t')\,\hat{x}_0/\sigma_d + \sin(t')\,\epsilon$$
    """
    return torch.cos(t_next) * x0_pred / sigma_data + torch.sin(t_next) * noise
