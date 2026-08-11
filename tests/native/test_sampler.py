r"""Replay the captured scheduler step against the owned solver at 1e-6."""

from __future__ import annotations

import json

import pytest

from tests.native._assert import assert_close_to_reference, load_reference


def test_sampler_step_replays_reference(reference_dir, fixture_cfg):
    pytest.importorskip("diffusers")
    from worldkernels.models.schedulers.flow_unipc import FlowUniPCMultistepScheduler

    manifest = json.loads((reference_dir / "manifest.json").read_text())
    if "sampler_step_x_t_next" not in manifest["tensors"]:
        pytest.skip("fixture set has no sampler_step capture")

    eps_pred = load_reference(reference_dir, "sampler_step_eps_pred")
    x_t = load_reference(reference_dir, "sampler_step_x_t")
    t = load_reference(reference_dir, "sampler_step_t")
    x_t_next = load_reference(reference_dir, "sampler_step_x_t_next")

    s = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0)
    s.set_timesteps(fixture_cfg.num_steps, device="cpu", shift=5.0)
    assert int(t.item()) == int(s.timesteps[0].item())

    prev_sample = s.step(eps_pred.float(), t, x_t.float(), return_dict=False)[0]
    assert_close_to_reference(prev_sample, x_t_next.float(), stage="sampler_step", name="step0")
