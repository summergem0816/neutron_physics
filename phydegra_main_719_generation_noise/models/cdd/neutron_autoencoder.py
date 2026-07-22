from typing import Any, Dict

import torch
from torch import nn

from .blindsr import LPE
from .blindsrlocal import IDASR, LPEL


class CDDContentEncoder(nn.Module):
    def __init__(self, dim: int, latent_dim: int, kernel_size: int):
        super().__init__()
        self.encoder = LPEL(dim=dim, out_dim=latent_dim, k=kernel_size)

    def forward(self, x):
        z_c, _ = self.encoder(x)
        return z_c, {"hq": x}


class CDDGlobalDegradationEncoder(nn.Module):
    def __init__(self, dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        self.encoder = LPE(dim=dim, out_dim=out_dim, k=kernel_size)

    def forward(self, x):
        out, _ = self.encoder(x)
        return out


class CDDLocalDegradationEncoder(nn.Module):
    def __init__(self, dim: int, out_dim: int, kernel_size: int):
        super().__init__()
        self.encoder = LPEL(dim=dim, out_dim=out_dim, k=kernel_size)

    def forward(self, x):
        out, _ = self.encoder(x)
        return out


class CDDDegradationCompressor(nn.Module):
    def __init__(self, global_dim: int, local_dim: int, latent_dim: int):
        super().__init__()
        self.global_to_map = nn.Sequential(
            nn.Linear(global_dim, local_dim, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(local_dim, local_dim, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(local_dim * 2, max(local_dim, latent_dim), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(max(local_dim, latent_dim), latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, global_code, local_code):
        global_map = self.global_to_map(global_code).unsqueeze(-1).unsqueeze(-1)
        global_map = global_map.expand(-1, -1, local_code.shape[-2], local_code.shape[-1])
        return self.fuse(torch.cat([global_map, local_code], dim=1))


class CDDDegradationExpander(nn.Module):
    def __init__(self, latent_dim: int, global_dim: int, local_dim: int):
        super().__init__()
        self.local_refine = nn.Sequential(
            nn.Conv2d(latent_dim, max(local_dim, latent_dim), kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(max(local_dim, latent_dim), local_dim, kernel_size=3, padding=1),
        )
        self.global_head = nn.Sequential(
            nn.Conv2d(latent_dim, global_dim, kernel_size=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, z_t):
        local_code = self.local_refine(z_t)
        global_code = self.global_head(z_t).flatten(1)
        return global_code, local_code


class CDDReconstructionDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        global_dim: int,
        local_dim: int,
        generator_kernel_size: int,
        n_feats_map: int,
    ):
        super().__init__()
        self.expander = CDDDegradationExpander(latent_dim, global_dim, local_dim)
        self.generator = IDASR(
            in_channels=in_dim,
            map_in_channels=local_dim,
            global_dim=global_dim,
            scale=1,
            kernel_size=generator_kernel_size,
            hf=False,
            n_feats_map=n_feats_map,
        )

    def forward(self, z_t, content_features):
        if isinstance(content_features, dict):
            clean = content_features["hq"]
        else:
            clean = content_features
        global_code, local_code = self.expander(z_t)
        return self.generator(clean, global_code, local_code)


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_dim,
        base_dim,
        latent_dim,
        use_skip,
        global_deg_dim=256,
        local_deg_dim=4,
        encoder_kernel_size=13,
        generator_kernel_size=3,
        n_feats_map=16,
    ):
        super().__init__()
        self.use_skip = True
        self.content_encoder = CDDContentEncoder(base_dim, latent_dim, encoder_kernel_size)
        self.global_degradation_encoder = CDDGlobalDegradationEncoder(base_dim, global_deg_dim, encoder_kernel_size)
        self.local_degradation_encoder = CDDLocalDegradationEncoder(base_dim, local_deg_dim, encoder_kernel_size)
        self.degradation_compressor = CDDDegradationCompressor(global_deg_dim, local_deg_dim, latent_dim)
        self.reconstruction_decoder = CDDReconstructionDecoder(
            in_dim=in_dim,
            latent_dim=latent_dim,
            global_dim=global_deg_dim,
            local_dim=local_deg_dim,
            generator_kernel_size=generator_kernel_size,
            n_feats_map=n_feats_map,
        )

        self.encoder = self.content_encoder
        self.decoder = self.reconstruction_decoder

    def encode_content(self, x):
        z_c, features = self.content_encoder(x)
        return {
            "z_c": z_c,
            "features": features,
            "clean_reference": x,
        }

    def encode_degradation(self, x):
        global_code = self.global_degradation_encoder(x)
        local_code = self.local_degradation_encoder(x)
        z_t = self.degradation_compressor(global_code, local_code)
        return {
            "z_t": z_t,
            "global_code": global_code,
            "local_code": local_code,
        }

    def reconstruct_from_content_degradation(self, z_t, content_features):
        return self.reconstruction_decoder(z_t, content_features)

    def forward(self, hq, lq=None):
        content = self.encode_content(hq)
        output: Dict[str, Any] = dict(content)
        if lq is not None:
            degradation = self.encode_degradation(lq)
            output.update(degradation)
            output["reconstruction"] = self.reconstruct_from_content_degradation(
                degradation["z_t"],
                content["features"],
            )
        return output

