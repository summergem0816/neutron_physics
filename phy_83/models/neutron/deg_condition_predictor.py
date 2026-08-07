from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        dtype = t.dtype
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -scale)
        angles = t[:, None] * freq[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DegConditionPredictor(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        clean_channels: int = 3,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.latent_stem = ConvBlock(latent_channels + 1, hidden_dim)
        self.latent_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, stride=1, padding=1),
        )

        fused_in = clean_channels + latent_channels + hidden_dim + 1
        self.image_stem = ConvBlock(fused_in, hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.blur_head = nn.Linear(hidden_dim, 4)
        self.flux_head = nn.Linear(hidden_dim, 1)

        self.scatter_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.readout_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        self.interaction_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        z_content: torch.Tensor,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_hw = z_content.shape[-2:]
        image_hw = clean.shape[-2:]

        time_embed = self.time_mlp(self.time_embed(t))
        latent_t = t.view(-1, 1, 1, 1).expand(-1, 1, latent_hw[0], latent_hw[1])
        latent_feat = self.latent_stem(torch.cat([z_content, latent_t], dim=1))

        time_bias = time_embed[:, :, None, None]
        latent_feat = latent_feat + time_bias
        pred_z = self.latent_head(latent_feat)

        latent_up = F.interpolate(pred_z, size=image_hw, mode="bilinear", align_corners=False)
        latent_feat_up = F.interpolate(latent_feat, size=image_hw, mode="bilinear", align_corners=False)
        image_t = t.view(-1, 1, 1, 1).expand(-1, 1, image_hw[0], image_hw[1])
        image_feat = self.image_stem(torch.cat([clean, latent_up, latent_feat_up, image_t], dim=1))
        image_feat = image_feat + time_bias

        global_feat = self.global_pool(image_feat).flatten(1)
        global_feat = self.global_head(global_feat)

        blur_params = torch.tanh(self.blur_head(global_feat))
        flux_delta = torch.tanh(self.flux_head(global_feat).squeeze(1))
        scatter_map = torch.sigmoid(self.scatter_head(image_feat))
        readout_map = torch.sigmoid(self.readout_head(image_feat))
        interaction_gate = torch.sigmoid(self.interaction_head(image_feat))

        return {
            "pred_z": pred_z,
            "blur_params": blur_params,
            "flux_delta": flux_delta,
            "scatter_map": scatter_map,
            "readout_map": readout_map,
            "interaction_gate": interaction_gate,
        }
