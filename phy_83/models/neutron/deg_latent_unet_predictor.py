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


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class DegLatentUNetPredictor(nn.Module):
    """Predict z_t with a small U-Net while keeping the z-only stage-2 API."""

    def __init__(
        self,
        latent_channels: int = 4,
        clean_channels: int = 3,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.clean_channels = int(clean_channels)
        self.hidden_dim = int(hidden_dim)

        base_dim = self.hidden_dim
        mid_dim = base_dim * 2
        deep_dim = base_dim * 4
        in_channels = self.latent_channels + self.clean_channels + 1

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim + mid_dim + deep_dim),
            nn.GELU(),
        )

        self.enc1 = ResidualConvBlock(in_channels, base_dim)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, mid_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.enc2 = ResidualConvBlock(mid_dim, mid_dim)

        self.down2 = nn.Sequential(
            nn.Conv2d(mid_dim, deep_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.enc3 = ResidualConvBlock(deep_dim, deep_dim)
        self.bottleneck = ResidualConvBlock(deep_dim, deep_dim)

        self.dec2 = ResidualConvBlock(deep_dim + mid_dim, mid_dim)
        self.dec1 = ResidualConvBlock(mid_dim + base_dim, base_dim)
        self.refine = ResidualConvBlock(base_dim, base_dim)
        self.head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, self.latent_channels, kernel_size=3, stride=1, padding=1),
        )

    @staticmethod
    def _add_time_bias(feat: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return feat + bias[:, :, None, None].to(device=feat.device, dtype=feat.dtype)

    def forward(
        self,
        z_content: torch.Tensor,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_hw = z_content.shape[-2:]
        clean_down = F.interpolate(clean, size=latent_hw, mode="bilinear", align_corners=False)
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, latent_hw[0], latent_hw[1])

        time_bias = self.time_mlp(self.time_embed(t))
        bias1, bias2, bias3 = torch.split(
            time_bias,
            [self.hidden_dim, self.hidden_dim * 2, self.hidden_dim * 4],
            dim=1,
        )

        x = torch.cat([z_content, clean_down, t_map], dim=1)
        enc1 = self._add_time_bias(self.enc1(x), bias1)
        enc2 = self._add_time_bias(self.enc2(self.down1(enc1)), bias2)
        deep = self._add_time_bias(self.enc3(self.down2(enc2)), bias3)
        deep = self.bottleneck(deep)

        up2 = F.interpolate(deep, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        pred_z = self.head(self.refine(dec1))
        return {"pred_z": pred_z}
