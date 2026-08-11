r"""adaLN-modulated video DiT blocks and the assembled backbone."""

from __future__ import annotations

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from worldkernels.models.components.attention import Attention, FrameKVCache
from worldkernels.models.components.embeddings import (
    PatchEmbed,
    SinusoidalTimesteps,
    TimestepEmbedding,
    VideoRoPE3D,
)
from worldkernels.models.components.normalization import RMSNorm

__all__ = ["FeedForward", "DiTBlock", "FinalLayer", "VideoDiT"]


class FeedForward(nn.Module):
    r"""GPT2-style two-layer MLP with GELU, no biases."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


def _adaln_modulation(x_dim: int, n_chunks: int, use_adaln_lora: bool, lora_dim: int) -> nn.Module:
    if use_adaln_lora:
        return nn.Sequential(
            nn.SiLU(),
            nn.Linear(x_dim, lora_dim, bias=False),
            nn.Linear(lora_dim, n_chunks * x_dim, bias=False),
        )
    return nn.Sequential(nn.SiLU(), nn.Linear(x_dim, n_chunks * x_dim, bias=False))


def _modulate(
    x: torch.Tensor, norm: nn.Module, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    r"""Self-attention, cross-attention, and MLP, each with adaLN(-LoRA) modulation.

    Modulation parameters are computed in float32 under the Wan strategy; the
    modulated streams run in the residual dtype. Residual update expressions
    keep the reference operand order for bit-comparable accumulation.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim, None, num_heads, x_dim // num_heads, use_wan_fp32_strategy=use_wan_fp32_strategy
        )
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.adaln_modulation_self_attn = _adaln_modulation(
            x_dim, 3, use_adaln_lora, adaln_lora_dim
        )
        self.adaln_modulation_cross_attn = _adaln_modulation(
            x_dim, 3, use_adaln_lora, adaln_lora_dim
        )
        self.adaln_modulation_mlp = _adaln_modulation(x_dim, 3, use_adaln_lora, adaln_lora_dim)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        kv_cache: FrameKVCache | None = None,
        frame_idx: int = 0,
        store_kv: bool = False,
    ) -> torch.Tensor:
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                mod_self = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                    3, dim=-1
                )
                mod_cross = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                    3, dim=-1
                )
                mod_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                    3, dim=-1
                )
            else:
                mod_self = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                mod_cross = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                mod_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_sa, scale_sa, gate_sa = (_expand(t) for t in mod_self)
        shift_ca, scale_ca, gate_ca = (_expand(t) for t in mod_cross)
        shift_mlp, scale_mlp, gate_mlp = (_expand(t) for t in mod_mlp)

        B, T, H, W, D = x_B_T_H_W_D.shape

        normalized = _modulate(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        result = rearrange(
            self.self_attn(
                rearrange(normalized, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                kv_cache=kv_cache,
                frame_idx=frame_idx,
                store_kv=store_kv,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * result

        normalized = _modulate(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        result = rearrange(
            self.cross_attn(rearrange(normalized, "b t h w d -> b (t h w) d"), crossattn_emb),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result * gate_ca + x_B_T_H_W_D

        normalized = _modulate(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * self.mlp(normalized)
        return x_B_T_H_W_D


class FinalLayer(nn.Module):
    r"""adaLN-modulated projection from model width to patch pixels."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
        )
        self.adaln_modulation = _adaln_modulation(hidden_size, 2, use_adaln_lora, adaln_lora_dim)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_wan_fp32_strategy:
            assert emb_B_T_D.dtype == torch.float32
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift, scale = (
                    self.adaln_modulation(emb_B_T_D)
                    + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
            shift = rearrange(shift, "b t d -> b t 1 1 d")
            scale = rearrange(scale, "b t d -> b t 1 1 d")
            x_B_T_H_W_D = _modulate(x_B_T_H_W_D, self.layer_norm, scale, shift)
            x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class VideoDiT(nn.Module):
    r"""adaLN-LoRA video diffusion transformer over patchified latents.

    The Cosmos-Predict2.5 backbone: patch embed with an appended padding-mask
    channel, 3D RoPE with NTK extrapolation, sinusoidal timesteps with a shared
    adaLN-LoRA stream, and a stack of `DiTBlock`. Subclasses add conditioning
    (actions, condition-mask channels) and causal per-frame execution.
    """

    def __init__(
        self,
        *,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.concat_padding_mask = concat_padding_mask
        self.use_adaln_lora = use_adaln_lora
        self.use_crossattn_projection = use_crossattn_projection
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels + 1 if concat_padding_mask else in_channels,
            out_channels=model_channels,
        )
        self.pos_embedder = VideoRoPE3D(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
        )
        self.t_embedder = nn.Sequential(
            SinusoidalTimesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        self.blocks = nn.ModuleList(
            DiTBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
                use_wan_fp32_strategy=use_wan_fp32_strategy,
            )
            for _ in range(num_blocks)
        )
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Patchify with the padding-mask channel appended; return tokens and RoPE table."""
        if self.concat_padding_mask:
            assert padding_mask is not None
            if padding_mask.shape[-2:] != x_B_C_T_H_W.shape[-2:]:
                padding_mask = F.interpolate(
                    padding_mask, size=x_B_C_T_H_W.shape[-2:], mode="nearest"
                )
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1),
                ],
                dim=1,
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        _, T, H, W, _ = x_B_T_H_W_D.shape
        rope_emb_L_1_1_D = self.pos_embedder(T, H, W)
        return x_B_T_H_W_D, rope_emb_L_1_1_D

    def compute_time_embeddings(
        self, timesteps_B_T: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Timestep embedding and adaLN-LoRA stream, float32 under the Wan strategy."""
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
        return t_embedding_B_T_D, adaln_lora_B_T_3D

    def project_context(self, crossattn_emb: torch.Tensor) -> torch.Tensor:
        if self.use_crossattn_projection:
            return self.crossattn_proj(crossattn_emb)
        return crossattn_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_B_T_H_W_D, rope_emb_L_1_1_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W, padding_mask=padding_mask
        )
        crossattn_emb = self.project_context(crossattn_emb)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.compute_time_embeddings(timesteps_B_T)

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
