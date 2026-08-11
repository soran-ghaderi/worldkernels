r"""CPU tests for the TrigFlow student sampling math."""

from __future__ import annotations

import math

import torch

from worldkernels.models.schedulers.trigflow import (
    TRIGFLOW_TIMES_4STEP,
    trigflow_scaling,
    trigflow_step,
    trigflow_time,
)


def test_sampling_times():
    assert TRIGFLOW_TIMES_4STEP == (
        math.pi / 2,
        math.atan(15.0),
        math.atan(5.0),
        math.atan(5.0 / 3.0),
    )
    assert all(a > b for a, b in zip(TRIGFLOW_TIMES_4STEP, TRIGFLOW_TIMES_4STEP[1:]))


def test_trigflow_time():
    assert trigflow_time(1e-4) == math.atan(1e-4)
    assert trigflow_time(3.0, sigma_data=2.0) == math.atan(1.5)


def test_scaling_matches_reference_formulas():
    t = torch.tensor(TRIGFLOW_TIMES_4STEP, dtype=torch.float64)
    c_skip, c_out, c_in, c_noise = trigflow_scaling(t, sigma_data=1.0)
    denom = torch.cos(t) + torch.sin(t)
    torch.testing.assert_close(c_skip, 1.0 / denom, rtol=0, atol=0)
    torch.testing.assert_close(c_in, 1.0 / denom, rtol=0, atol=0)
    torch.testing.assert_close(c_out, -torch.sin(t) / denom, rtol=0, atol=0)
    torch.testing.assert_close(c_noise, torch.sin(t) / denom, rtol=0, atol=0)


def test_scaling_limits_and_dtype():
    t0 = torch.tensor([0.0], dtype=torch.bfloat16)
    c_skip, c_out, c_in, c_noise = trigflow_scaling(t0)
    assert c_skip.dtype == torch.bfloat16
    torch.testing.assert_close(c_skip.float(), torch.ones(1), rtol=0, atol=0)
    torch.testing.assert_close(c_out.float(), torch.zeros(1), rtol=0, atol=0)
    torch.testing.assert_close(c_noise.float(), torch.zeros(1), rtol=0, atol=0)
    assert c_in.shape == t0.shape


def test_perfect_velocity_reconstructs_x0():
    r"""The sCM identity \(c_\text{skip} x_t + c_\text{out} F = x_0\) for a perfect net.

    ``c_in`` maps the TrigFlow sample \(x_t = \cos(t) x_0 + \sin(t) \epsilon\)
    onto the rectified-flow path at \(\sigma_f = \sin t / (\cos t + \sin t)\),
    where the teacher-parameterized net predicts \(v = \epsilon - x_0\); the
    reconstruction must then recover \(x_0\) exactly.
    """
    g = torch.Generator().manual_seed(11)
    x0 = torch.randn(2, 4, 1, 3, 3, generator=g, dtype=torch.float64)
    eps = torch.randn(2, 4, 1, 3, 3, generator=g, dtype=torch.float64)

    for t_val in TRIGFLOW_TIMES_4STEP[1:]:
        t = torch.tensor(t_val, dtype=torch.float64)
        x_t = torch.cos(t) * x0 + torch.sin(t) * eps
        c_skip, c_out, c_in, _ = trigflow_scaling(t)
        sigma_f = torch.sin(t) / (torch.cos(t) + torch.sin(t))
        flow_sample = c_in * x_t
        torch.testing.assert_close(flow_sample, (1 - sigma_f) * x0 + sigma_f * eps)
        recon = c_skip * x_t + c_out * (eps - x0)
        torch.testing.assert_close(recon, x0, rtol=1e-12, atol=1e-12)


def test_step_renoises_to_next_angle():
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(1, 4, 1, 2, 2, generator=g)
    noise = torch.randn(1, 4, 1, 2, 2, generator=g)
    t_next = torch.tensor(TRIGFLOW_TIMES_4STEP[1])
    out = trigflow_step(x0, noise, t_next)
    torch.testing.assert_close(out, torch.cos(t_next) * x0 + torch.sin(t_next) * noise)
    torch.testing.assert_close(trigflow_step(x0, noise, torch.tensor(0.0)), x0)
