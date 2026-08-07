import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_kernel(kernel: torch.Tensor) -> torch.Tensor:
    kernel = torch.clamp(kernel, min=0.0)
    return kernel / kernel.sum().clamp_min(1e-8)


def gaussian_kernel2d(
    kernel_size: int,
    sigma_x,
    sigma_y,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma_x = torch.as_tensor(sigma_x, device=device, dtype=dtype).clamp_min(1e-6)
    sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype).clamp_min(1e-6)
    kernel = torch.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
    return _normalize_kernel(kernel)


def _conv_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    weight = kernel[None, None, :, :].to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, weight, padding=padding, groups=channels)


class NeutronPhysicalForward(nn.Module):
    def __init__(
        self,
        geo_kernel_size: int = 21,
        det_kernel_size: int = 15,
        scatter_kernel_size: int = 81,
        theta_div_deg: float = 1.0,
        object_detector_distance_mm: float = 50.0,
        detector_pitch_mm: float = 0.66,
        detector_pixel_width_mm: float = 0.6,
        scintillator_sigma_px_init: float = 0.35,
        flux_ref_max: float = 3.0e8,
        particle_count_min: float = 1.0e7,
        particle_count_max: float = 3.0e8,
        scatter_strength_max: float = 0.08,
        readout_sigma_max: float = 0.01,
        latent_modulation_scale: float = 0.15,
        latent_channels: int = 4,
        trainable_blur: bool = True,
        trainable_noise_heads: bool = True,
    ):
        super().__init__()
        self.geo_kernel_size = geo_kernel_size
        self.det_kernel_size = det_kernel_size
        self.scatter_kernel_size = scatter_kernel_size
        self.detector_pitch_mm = detector_pitch_mm
        self.flux_ref_max = flux_ref_max
        self.particle_count_min = particle_count_min
        self.particle_count_max = particle_count_max
        self.scatter_strength_max = scatter_strength_max
        self.readout_sigma_max = readout_sigma_max
        self.latent_modulation_scale = latent_modulation_scale

        sigma_geo_mm = object_detector_distance_mm * math.tan(math.radians(theta_div_deg))
        sigma_geo_px = sigma_geo_mm / detector_pitch_mm
        sigma_det_px = math.sqrt((detector_pixel_width_mm / math.sqrt(12.0) / detector_pitch_mm) ** 2 + scintillator_sigma_px_init ** 2)

        self.log_sigma_geo = nn.Parameter(torch.tensor(math.log(max(sigma_geo_px, 1e-4))), requires_grad=trainable_blur)
        self.log_sigma_det = nn.Parameter(torch.tensor(math.log(max(sigma_det_px, 1e-4))), requires_grad=trainable_blur)

        self.scatter_sigma_px = max(8.0, scatter_kernel_size / 6.0)

        self.latent_to_flux = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.latent_to_scatter = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.latent_to_readout = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        if not trainable_noise_heads:
            for module in (self.latent_to_flux, self.latent_to_scatter, self.latent_to_readout):
                for param in module.parameters():
                    param.requires_grad = False

    def _base_particle_count(self, t: torch.Tensor) -> torch.Tensor:
        return self.particle_count_max - (self.particle_count_max - self.particle_count_min) * t

    def _latent_scalar(self, head: nn.Module, degradation_latent: torch.Tensor | None, batch_size: int, device, dtype) -> torch.Tensor:
        if degradation_latent is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)
        value = head(degradation_latent).flatten(1).squeeze(1)
        return torch.tanh(value)

    def _effective_particle_count(self, t: torch.Tensor, degradation_latent: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = t.shape[0]
        delta = self._latent_scalar(self.latent_to_flux, degradation_latent, batch_size, t.device, t.dtype)
        base_count = self._base_particle_count(t)
        effective_count = base_count * torch.exp(self.latent_modulation_scale * delta)
        return base_count, effective_count.clamp_min(1.0)

    def _sigma_geo(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_geo).clamp_min(1e-4)

    def _sigma_det(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_det).clamp_min(1e-4)

    def _apply_blur(self, x: torch.Tensor, sigma: torch.Tensor, kernel_size: int) -> torch.Tensor:
        outputs = []
        for idx in range(x.shape[0]):
            kernel = gaussian_kernel2d(
                kernel_size,
                sigma_x=sigma[idx],
                sigma_y=sigma[idx],
                device=x.device,
                dtype=x.dtype,
            )
            outputs.append(_conv_same(x[idx:idx + 1], kernel))
        return torch.cat(outputs, dim=0)

    def apply_source_flux(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transmission = clean.clamp(-1.0, 1.0).add(1.0).div(2.0)
        base_count, effective_count = self._effective_particle_count(t, degradation_latent)
        x_s = transmission
        return x_s, {
            "transmission": transmission,
            "particle_count_base": base_count,
            "particle_count_effective": effective_count,
            "source_gain": (effective_count / self.flux_ref_max),
        }

    def apply_geometric_blur(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.ones((x.shape[0],), device=x.device, dtype=x.dtype) * self._sigma_geo().to(device=x.device, dtype=x.dtype)
        return self._apply_blur(x, sigma, self.geo_kernel_size), sigma

    def apply_detector_blur(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.ones((x.shape[0],), device=x.device, dtype=x.dtype) * self._sigma_det().to(device=x.device, dtype=x.dtype)
        return self._apply_blur(x, sigma, self.det_kernel_size), sigma

    def apply_scatter_background(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = gaussian_kernel2d(
            self.scatter_kernel_size,
            sigma_x=self.scatter_sigma_px,
            sigma_y=self.scatter_sigma_px,
            device=x.device,
            dtype=x.dtype,
        )
        scatter_base = _conv_same(x, kernel)
        delta = self._latent_scalar(self.latent_to_scatter, degradation_latent, x.shape[0], x.device, x.dtype)
        particle_ratio = self._base_particle_count(t) / self.particle_count_max
        scarcity = 1.0 - particle_ratio
        strength = self.scatter_strength_max * scarcity + self.latent_modulation_scale * 0.5 * delta
        strength = strength.clamp(min=0.0).view(-1, 1, 1, 1)
        return strength * scatter_base, strength.squeeze(-1).squeeze(-1).squeeze(-1)

    def apply_poisson_layer(
        self,
        x: torch.Tensor,
        effective_particle_count: torch.Tensor,
        sample_noise: bool = True,
    ) -> torch.Tensor:
        flux = effective_particle_count.view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        expected_counts = torch.clamp(x, min=0.0) * flux
        sampled = torch.poisson(expected_counts) if sample_noise else expected_counts
        return sampled / flux.clamp_min(1e-8)

    def apply_readout_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None,
        sample_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self._latent_scalar(self.latent_to_readout, degradation_latent, x.shape[0], x.device, x.dtype)
        particle_ratio = self._base_particle_count(t) / self.particle_count_max
        scarcity = 1.0 - particle_ratio
        sigma = (self.readout_sigma_max * scarcity + self.latent_modulation_scale * 0.1 * delta).clamp(min=0.0)
        sigma_view = sigma.view(-1, 1, 1, 1)
        if not sample_noise:
            return x, sigma
        return x + torch.randn_like(x) * sigma_view, sigma

    def forward(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None = None,
        sample_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        source_out, source_stats = self.apply_source_flux(clean, t, degradation_latent)
        geo_out, sigma_geo = self.apply_geometric_blur(source_out)
        det_out, sigma_det = self.apply_detector_blur(geo_out)
        scatter, scatter_strength = self.apply_scatter_background(det_out, t, degradation_latent)
        pre_poisson = det_out + scatter
        poisson_out = self.apply_poisson_layer(
            pre_poisson,
            source_stats["particle_count_effective"],
            sample_noise=sample_noise,
        )
        noisy_out, readout_sigma = self.apply_readout_noise(poisson_out, t, degradation_latent, sample_noise=sample_noise)
        degraded = noisy_out.clamp(0.0, 1.0).mul(2.0).sub(1.0)

        return {
            "transmission": source_stats["transmission"],
            "source": source_out,
            "geo_blur": geo_out,
            "det_blur": det_out,
            "scatter": scatter,
            "pre_poisson": pre_poisson,
            "degraded": degraded,
            "particle_count_base": source_stats["particle_count_base"],
            "particle_count_effective": source_stats["particle_count_effective"],
            "source_gain": source_stats["source_gain"],
            "sigma_geo": sigma_geo,
            "sigma_det": sigma_det,
            "scatter_strength": scatter_strength,
            "readout_sigma": readout_sigma,
            "t": t,
        }
