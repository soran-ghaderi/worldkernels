r"""Owned flow-matching UniPC multistep solver.

Full port of the rectified-flow UniPC solver used by Wan and Cosmos-Predict2.5
teacher models (UniPC B(h), predictor-corrector, ``predict_x0`` form on the
flow parameterization \(x_t = (1-\sigma) x_0 + \sigma \epsilon\), velocity
prediction \(v = \epsilon - x_0\)). Owning every line of the solver is what
lets the native DreamDojo pipeline match golden fixtures at 1e-6.

Unlike the upstream copy, the ``shift`` warp
\(\sigma \mapsto s\sigma / (1 + (s-1)\sigma)\) is applied exactly once, in
``set_timesteps``; upstream applied it a second time at construction, which is
the identity for the teacher's ``shift=1`` construction.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

__all__ = ["FlowUniPCMultistepScheduler"]


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):  # type: ignore[misc]
    r"""UniPC multistep solver for flow-matching video diffusion models."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        shift: float = 1.0,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] | None = None,
    ) -> None:
        if solver_type not in ("bh1", "bh2"):
            raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)

        self._num_train_timesteps = num_train_timesteps
        self._solver_order = solver_order
        self._shift = shift
        self._solver_type = solver_type
        self._lower_order_final = lower_order_final

        self.num_inference_steps: int | None = None
        self.sigmas = sigmas.to("cpu")
        self.timesteps = sigmas * num_train_timesteps
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.model_outputs: list[torch.Tensor | None] = [None] * solver_order
        self.timestep_list: list[int | torch.Tensor | None] = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = list(disable_corrector) if disable_corrector else []
        self.last_sample: torch.Tensor | None = None
        self.this_order = solver_order
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @classmethod
    def for_wan(
        cls,
        *,
        flow_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ) -> "FlowUniPCMultistepScheduler":
        r"""Build a scheduler with Wan's flow-matching defaults.

        Args:
            flow_shift: Timestep-schedule shift \(s\), applied in
                ``set_timesteps`` as \(\sigma \mapsto s\sigma/(1+(s-1)\sigma)\);
                larger \(s\) spends more steps at high noise. Wan uses ~3.0 at
                480p, ~5.0 at 720p.
            num_train_timesteps: Training timestep count.
        """
        return cls(num_train_timesteps=num_train_timesteps, shift=flow_shift)

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        shift: float | None = None,
        use_kerras_sigma: bool = False,
    ) -> None:
        r"""Set the discrete sigma/timestep schedule for inference.

        Args:
            num_inference_steps: Number of solver steps.
            device: Device for the integer timestep tensor.
            shift: Schedule warp \(s\); defaults to the construction-time value.
            use_kerras_sigma: Use the EDM Karras sigma ladder mapped to flow
                sigmas via \(\sigma/(1+\sigma)\) instead of the shifted linspace.
        """
        if use_kerras_sigma:
            sigma_max, sigma_min, rho = 200, 0.01, 7
            ramp = np.arange(num_inference_steps + 1) / num_inference_steps
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            sigmas = sigmas / (1 + sigmas)
        else:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]
            if shift is None:
                shift = self._shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        timesteps = sigmas * self._num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self._solver_order
        self.timestep_list = [None] * self._solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def convert_model_output(
        self, model_output: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        r"""Convert velocity prediction to the \(x_0\) prediction UniPC integrates."""
        sigma_t = self.sigmas[self.step_index]
        return sample - sigma_t * model_output

    def multistep_uni_p_bh_update(self, sample: torch.Tensor, order: int) -> torch.Tensor:
        r"""UniP predictor update of the given order from stored model outputs."""
        m0 = self.model_outputs[-1]
        assert m0 is not None
        x = sample
        device = sample.device

        assert self._step_index is not None
        sigma_t, sigma_s0 = self.sigmas[self._step_index + 1], self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list[torch.Tensor | float] = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - i
            mi = self.model_outputs[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks_t = torch.tensor(rks, device=device)

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h * phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = hh if self._solver_type == "bh1" else torch.expm1(hh)

        R = []
        b = []
        for i in range(1, order + 1):
            R.append(torch.pow(rks_t, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R_t = torch.stack(R)
        b_t = torch.tensor(b, device=device)

        if D1s:
            D1s_t = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R_t[:-1, :-1], b_t[:-1]).to(device).to(x.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s_t)
        else:
            pred_res = torch.zeros_like(x)

        x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0 - alpha_t * B_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        r"""UniC corrector update using the model output at the predicted point."""
        m0 = self.model_outputs[-1]
        assert m0 is not None
        x = last_sample
        model_t = this_model_output
        device = this_sample.device

        assert self._step_index is not None
        sigma_t, sigma_s0 = self.sigmas[self._step_index], self.sigmas[self._step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list[torch.Tensor | float] = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - (i + 1)
            mi = self.model_outputs[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks_t = torch.tensor(rks, device=device)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = hh if self._solver_type == "bh1" else torch.expm1(hh)

        R = []
        b = []
        for i in range(1, order + 1):
            R.append(torch.pow(rks_t, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R_t = torch.stack(R)
        b_t = torch.tensor(b, device=device)

        D1s_t = torch.stack(D1s, dim=1) if D1s else None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R_t, b_t).to(device).to(x.dtype)

        if D1s_t is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s_t)
        else:
            corr_res = torch.zeros_like(x)
        D1_t = model_t - m0
        x_t = (
            sigma_t / sigma_s0 * x
            - alpha_t * h_phi_1 * m0
            - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        )
        return x_t.to(x.dtype)

    def index_for_timestep(
        self, timestep: int | torch.Tensor, schedule_timesteps: torch.Tensor | None = None
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        if self._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput | tuple[torch.Tensor, torch.Tensor]:
        r"""Advance the sample one solver step given the velocity prediction."""
        del generator
        if self.num_inference_steps is None:
            raise ValueError("run 'set_timesteps' before calling 'step'")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None

        use_corrector = (
            self._step_index > 0
            and self._step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            assert self.last_sample is not None
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self._solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self._lower_order_final:
            this_order = min(self._solver_order, len(self.timesteps) - self._step_index)
        else:
            this_order = self._solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(sample=sample, order=self.this_order)

        if self.lower_order_nums < self._solver_order:
            self.lower_order_nums += 1
        self._step_index += 1

        if not return_dict:
            return (prev_sample, model_output_convert)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward-diffuse samples to the noise levels of ``timesteps``."""
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        if self._begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self._step_index is not None:
            step_indices = [self._step_index] * timesteps.shape[0]
        else:
            step_indices = [self._begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self) -> int:
        return int(self._num_train_timesteps)
