r"""CPU tests for the owned flow-matching UniPC solver.

Golden literals were captured from the reference solver
(`cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py`) constructed as the
DreamDojo teacher does: ``(num_train_timesteps=1000, shift=1)`` then
``set_timesteps(8, shift=5.0)``, driven by a deterministic synthetic velocity
field. The owned port matched bit-exactly at capture time; the test budget is
1e-6 to absorb BLAS/platform variance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldkernels.models.schedulers.flow_unipc import FlowUniPCMultistepScheduler

GOLDEN_TIMESTEPS = [999, 972, 937, 892, 833, 749, 624, 416]
GOLDEN_SIGMAS_FIRST5 = [0.999799848, 0.972006023, 0.937265456, 0.892601848, 0.833055377]
GOLDEN_TRAJECTORY = {
    0: [-0.144531727, 0.781717718, 0.941713333, -1.106578827],
    3: [-0.135584623, 0.757901073, 0.914444089, -1.070225358],
    7: [-0.136135429, 0.575011969, 0.709882677, -0.912395954],
}


def _teacher_scheduler(num_steps: int = 8) -> FlowUniPCMultistepScheduler:
    s = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0)
    s.set_timesteps(num_steps, device="cpu", shift=5.0)
    return s


def _base_linspace(n: int) -> np.ndarray:
    r"""Sigma linspace exactly as the solver builds it (float32-rounded extremes)."""
    init_sigmas = (1.0 - np.linspace(1, 1 / 1000, 1000)[::-1]).astype(np.float32)
    return np.linspace(float(init_sigmas[0]), float(init_sigmas[-1]), n + 1)[:-1]


def test_schedule_closed_form():
    s = _teacher_scheduler(35)
    base = _base_linspace(35)
    expected = 5.0 * base / (1.0 + 4.0 * base)
    np.testing.assert_allclose(s.sigmas[:-1].numpy(), expected.astype(np.float32), rtol=0, atol=0)
    assert s.sigmas[-1].item() == 0.0
    assert s.timesteps.dtype == torch.int64
    np.testing.assert_array_equal(s.timesteps.numpy(), (expected * 1000).astype(np.int64))


def test_golden_trajectory_matches_reference():
    s = _teacher_scheduler(8)
    assert s.timesteps.tolist() == GOLDEN_TIMESTEPS
    np.testing.assert_allclose(s.sigmas[:5].numpy(), GOLDEN_SIGMAS_FIRST5, rtol=0, atol=1e-9)

    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 1, 1, 2, 2, generator=g, dtype=torch.float32)
    for i, t in enumerate(s.timesteps):
        v = torch.tanh(x) * 0.3 + torch.sin(torch.tensor(float(t)) / 250.0) * 0.05
        x = s.step(v, t, x, return_dict=False)[0]
        if i in GOLDEN_TRAJECTORY:
            np.testing.assert_allclose(
                x.flatten().numpy(), GOLDEN_TRAJECTORY[i], rtol=1e-6, atol=1e-6
            )


def test_constant_velocity_integrates_exactly():
    r"""For \(v = \epsilon - x_0\) the flow ODE is linear; UniPC must land on \(x_0\)."""
    s = _teacher_scheduler(8)
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn(1, 4, 2, 4, 4, generator=g)
    eps = torch.randn(1, 4, 2, 4, 4, generator=g)
    v = eps - x0

    sigma0 = s.sigmas[0]
    x = (1 - sigma0) * x0 + sigma0 * eps
    for t in s.timesteps:
        x = s.step(v, t, x, return_dict=False)[0]
    torch.testing.assert_close(x, x0, rtol=1e-5, atol=1e-5)


def test_order_ramp_and_step_index():
    s = _teacher_scheduler(4)
    x = torch.zeros(1, 1, 1, 2, 2)
    assert s.step_index is None
    s.step(torch.ones_like(x), s.timesteps[0], x, return_dict=False)
    assert s.step_index == 1
    assert s.this_order == 1
    s.step(torch.ones_like(x), s.timesteps[1], x, return_dict=False)
    assert s.this_order == 2


def test_step_before_set_timesteps_raises():
    s = FlowUniPCMultistepScheduler()
    with pytest.raises(ValueError, match="set_timesteps"):
        s.step(torch.zeros(1), 999, torch.zeros(1))


def test_for_wan_applies_single_shift_warp():
    s = FlowUniPCMultistepScheduler.for_wan(flow_shift=5.0)
    s.set_timesteps(8, device="cpu")
    base = _base_linspace(8)
    expected = 5.0 * base / (1.0 + 4.0 * base)
    np.testing.assert_allclose(s.sigmas[:-1].numpy(), expected.astype(np.float32), rtol=0, atol=0)
    assert s.sigmas[-1].item() == 0.0
    diffs = np.diff(s.sigmas.numpy())
    assert (diffs < 0).all()


def test_kerras_sigma_ladder():
    r"""The Karras ladder keeps its n+1 rungs as timesteps (reference quirk)."""
    s = FlowUniPCMultistepScheduler()
    s.set_timesteps(6, device="cpu", use_kerras_sigma=True)
    sig = s.sigmas.numpy()
    assert sig[0] == pytest.approx(200 / 201, abs=1e-6)
    assert (np.diff(sig) < 0).all()
    assert len(sig) == 8
    assert s.num_inference_steps == 7


def test_add_noise_matches_flow_interpolation():
    s = _teacher_scheduler(4)
    x0 = torch.full((2, 1, 1, 1, 1), 2.0)
    eps = torch.full((2, 1, 1, 1, 1), -1.0)
    t = s.timesteps[:2]
    noisy = s.add_noise(x0, eps, t)
    sig = s.sigmas[:2].view(2, 1, 1, 1, 1)
    torch.testing.assert_close(noisy, (1 - sig) * x0 + sig * eps)
