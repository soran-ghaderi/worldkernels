r"""Action-conditioned video DiT for DreamDojo (teacher clip + causal student paths)."""

from __future__ import annotations

from typing import Any

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from worldkernels.models.components.attention import FrameKVCache
from worldkernels.models.components.dit import VideoDiT

__all__ = ["ActionEmbedder", "ActionVideoDiT", "NET_2B", "NET_14B"]

NET_2B: dict[str, Any] = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    use_wan_fp32_strategy=True,
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
)

NET_14B: dict[str, Any] = {
    **NET_2B,
    "model_channels": 5120,
    "num_heads": 40,
    "num_blocks": 36,
}


class ActionEmbedder(nn.Module):
    r"""Two-layer GELU(tanh) MLP projecting flattened per-frame action chunks."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class ActionVideoDiT(VideoDiT):
    r"""DreamDojo DiT: per-latent-frame action chunks injected into the adaLN stream.

    Actions arrive as ``(B, T_a, action_dim)`` with
    ``T_a = num_action_per_latent_frame * generated_frames``; each latent
    frame's chunk is flattened and projected by two MLPs added to the timestep
    embedding and the adaLN-LoRA stream (zero contribution for the conditioning
    frame). An extra input channel carries the condition-video mask. `forward`
    runs the bidirectional teacher clip; `forward_frame` runs one latent frame
    causally against a per-block `FrameKVCache` with absolute-position RoPE.
    """

    def __init__(
        self,
        *,
        action_dim: int = 384,
        num_action_per_latent_frame: int = 4,
        hidden_dim_in_action_embedder: int | None = None,
        timestep_scale: float = 0.001,
        in_channels: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels=in_channels + 1, **kwargs)
        self.action_dim = action_dim
        self.num_action_per_latent_frame = num_action_per_latent_frame
        self.timestep_scale = timestep_scale
        hidden = hidden_dim_in_action_embedder or self.model_channels * 4
        self.action_embedder_B_D = ActionEmbedder(
            action_dim * num_action_per_latent_frame, hidden, self.model_channels
        )
        self.action_embedder_B_3D = ActionEmbedder(
            action_dim * num_action_per_latent_frame, hidden, 3 * self.model_channels
        )

    def _embed_action_chunks(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Project per-latent-frame action chunks, zero-padded for the conditioning frame."""
        num_actions = action.shape[1]
        action = rearrange(
            action, "b (t a) d -> b t (a d)", t=num_actions // self.num_action_per_latent_frame
        )
        action_emb_B_D = self.action_embedder_B_D(action)
        action_emb_B_3D = self.action_embedder_B_3D(action)
        zero_pad_B_D = torch.zeros_like(action_emb_B_D[:, :1, :])
        zero_pad_B_3D = torch.zeros_like(action_emb_B_3D[:, :1, :])
        return (
            torch.cat([zero_pad_B_D, action_emb_B_D], dim=1),
            torch.cat([zero_pad_B_3D, action_emb_B_3D], dim=1),
        )

    def forward(  # type: ignore[override]
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert condition_video_input_mask_B_C_T_H_W is not None
        assert action is not None
        x_B_C_T_H_W = torch.cat(
            [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1
        )
        timesteps_B_T = timesteps_B_T * self.timestep_scale
        action_emb_B_D, action_emb_B_3D = self._embed_action_chunks(action)

        x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W, padding_mask=padding_mask
        )
        crossattn_emb = self.project_context(crossattn_emb)

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )

        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D
        )
        return self.unpatchify(x_B_T_H_W_O)

    def forward_frame(
        self,
        x_B_C_1_H_W: torch.Tensor,
        frame_idx: int,
        timesteps_B_1: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_1_H_W: torch.Tensor,
        action: torch.Tensor | None = None,
        kv_caches: list[FrameKVCache] | None = None,
        store_kv: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Denoise one latent frame at absolute ``frame_idx`` against cached context.

        Args:
            action: The single frame's action chunk
                ``(B, num_action_per_latent_frame, action_dim)``; ``None`` for
                the conditioning frame (zero action contribution).
            kv_caches: One `FrameKVCache` per block; read under a rolling
                window, written only when ``store_kv`` is set.
        """
        x_B_C_1_H_W = torch.cat(
            [x_B_C_1_H_W, condition_video_input_mask_B_C_1_H_W.type_as(x_B_C_1_H_W)], dim=1
        )
        timesteps_B_1 = timesteps_B_1 * self.timestep_scale

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_1.ndim == 1:
                timesteps_B_1 = timesteps_B_1.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_1)
            if action is not None:
                action_flat = rearrange(action, "b t d -> b 1 (t d)")
                t_embedding_B_T_D = t_embedding_B_T_D + self.action_embedder_B_D(action_flat)
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + self.action_embedder_B_3D(action_flat)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
            if self.use_wan_fp32_strategy:
                t_embedding_B_T_D = t_embedding_B_T_D.float()

        x_B_T_H_W_D, _ = self.prepare_embedded_sequence(x_B_C_1_H_W, padding_mask=padding_mask)
        _, T, H, W, _ = x_B_T_H_W_D.shape
        assert T == 1, "forward_frame expects a single latent frame"
        crossattn_emb = self.project_context(crossattn_emb)
        rope_emb_L_1_1_D = self.pos_embedder.forward_frame(frame_idx, H, W)

        for i, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                kv_cache=kv_caches[i] if kv_caches is not None else None,
                frame_idx=frame_idx,
                store_kv=store_kv,
            )

        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D
        )
        return self.unpatchify(x_B_T_H_W_O)

    def allocate_kv_caches(
        self,
        *,
        max_frames: int,
        batch_size: int,
        latent_h: int,
        latent_w: int,
        window: int | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[FrameKVCache]:
        r"""One rolling KV cache per block sized for ``latent_h x latent_w`` frames."""
        tokens = (latent_h // self.patch_spatial) * (latent_w // self.patch_spatial)
        head_dim = self.model_channels // self.num_heads
        return [
            FrameKVCache.allocate(
                max_frames=max_frames,
                batch_size=batch_size,
                tokens_per_frame=tokens,
                num_heads=self.num_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
                window=window,
            )
            for _ in range(len(self.blocks))
        ]
