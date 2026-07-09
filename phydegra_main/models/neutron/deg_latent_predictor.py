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


class DegLatentPredictor(nn.Module):
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

        in_channels = latent_channels + clean_channels + 1
        self.stem = ConvBlock(in_channels, hidden_dim)
        self.res_block = ConvBlock(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(
        self,
        z_content: torch.Tensor,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_hw = z_content.shape[-2:]
        clean_down = F.interpolate(clean, size=latent_hw, mode="bilinear", align_corners=False)
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, latent_hw[0], latent_hw[1])

        feat = self.stem(torch.cat([z_content, clean_down, t_map], dim=1))
        time_bias = self.time_mlp(self.time_embed(t))[:, :, None, None]
        feat = feat + time_bias
        feat = feat + self.res_block(feat)
        pred_z = self.head(feat)
        return {"pred_z": pred_z}
