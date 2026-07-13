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


class PhysicalConditionEncoder(nn.Module):
    def __init__(self, latent_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_channels + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.blur_head = nn.Linear(hidden_dim, 4)
        self.flux_head = nn.Linear(hidden_dim, 1)

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(latent_channels + 1, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
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

    def forward(self, degradation_latent: torch.Tensor | None, t: torch.Tensor, output_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        batch = t.shape[0]
        device = t.device
        dtype = t.dtype

        if degradation_latent is None:
            pooled = torch.zeros(batch, 0, device=device, dtype=dtype)
            blur_params = torch.zeros(batch, 4, device=device, dtype=dtype)
            flux_delta = torch.zeros(batch, device=device, dtype=dtype)
            scatter_map = torch.zeros(batch, 1, output_hw[0], output_hw[1], device=device, dtype=dtype)
            readout_map = torch.zeros_like(scatter_map)
            interaction_gate = torch.zeros_like(scatter_map)
            return {
                "blur_params": blur_params,
                "flux_delta": flux_delta,
                "scatter_map": scatter_map,
                "readout_map": readout_map,
                "interaction_gate": interaction_gate,
            }

        latent_up = F.interpolate(degradation_latent, size=output_hw, mode="bilinear", align_corners=False)
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, output_hw[0], output_hw[1])
        spatial_input = torch.cat([latent_up, t_map], dim=1)
        spatial_feat = self.spatial_stem(spatial_input)

        pooled = self.global_pool(degradation_latent).flatten(1)
        global_input = torch.cat([pooled, t.view(-1, 1)], dim=1)
        global_feat = self.global_mlp(global_input)

        blur_params = torch.tanh(self.blur_head(global_feat))
        flux_delta = torch.tanh(self.flux_head(global_feat).squeeze(1))
        scatter_map = torch.sigmoid(self.scatter_head(spatial_feat))
        readout_map = torch.sigmoid(self.readout_head(spatial_feat))
        interaction_gate = torch.sigmoid(self.interaction_head(spatial_feat))
        return {
            "blur_params": blur_params,
            "flux_delta": flux_delta,
            "scatter_map": scatter_map,
            "readout_map": readout_map,
            "interaction_gate": interaction_gate,
        }


class BlurInteractionBlock(nn.Module):
    def __init__(self, image_channels: int, latent_channels: int, hidden_dim: int = 32):
        super().__init__()
        in_channels = image_channels * 2 + latent_channels + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, image_channels, kernel_size=3, stride=1, padding=1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        geo_out: torch.Tensor,
        det_out: torch.Tensor,
        degradation_latent: torch.Tensor | None,
        t: torch.Tensor,
        interaction_gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = det_out.shape
        if degradation_latent is None:
            latent_up = torch.zeros(batch, 0, height, width, device=det_out.device, dtype=det_out.dtype)
        else:
            latent_up = F.interpolate(degradation_latent, size=(height, width), mode="bilinear", align_corners=False)
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, height, width)
        feat = torch.cat([geo_out, det_out, latent_up, t_map], dim=1)
        residual = self.net(feat)
        if interaction_gate is not None:
            residual = residual * interaction_gate
        out = det_out + self.res_scale.to(device=det_out.device, dtype=det_out.dtype) * residual
        return out, residual


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
        shared_channel_stochastic: bool = False,
        force_grayscale: bool = False,
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
        self.shared_channel_stochastic = shared_channel_stochastic
        self.force_grayscale = force_grayscale

        sigma_geo_mm = object_detector_distance_mm * math.tan(math.radians(theta_div_deg))
        sigma_geo_px = sigma_geo_mm / detector_pitch_mm
        sigma_det_px = math.sqrt((detector_pixel_width_mm / math.sqrt(12.0) / detector_pitch_mm) ** 2 + scintillator_sigma_px_init ** 2)

        self.log_sigma_geo = nn.Parameter(torch.tensor(math.log(max(sigma_geo_px, 1e-4))), requires_grad=trainable_blur)
        self.log_sigma_det = nn.Parameter(torch.tensor(math.log(max(sigma_det_px, 1e-4))), requires_grad=trainable_blur)

        self.scatter_sigma_px = max(8.0, scatter_kernel_size / 6.0)
        self.blur_modulation_limit = 0.2

        self.condition_encoder = PhysicalConditionEncoder(latent_channels=latent_channels)
        self.blur_interaction = BlurInteractionBlock(image_channels=3, latent_channels=latent_channels)

        if not trainable_noise_heads:
            for module in (self.condition_encoder, self.blur_interaction):
                for param in module.parameters():
                    param.requires_grad = False

    def _base_particle_count(self, t: torch.Tensor) -> torch.Tensor:
        return self.particle_count_max - (self.particle_count_max - self.particle_count_min) * t

    def _sigma_geo(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_geo).clamp_min(1e-4)

    def _sigma_det(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_det).clamp_min(1e-4)

    def _apply_blur(self, x: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor, kernel_size: int) -> torch.Tensor:
        outputs = []
        for idx in range(x.shape[0]):
            kernel = gaussian_kernel2d(
                kernel_size,
                sigma_x=sigma_x[idx],
                sigma_y=sigma_y[idx],
                device=x.device,
                dtype=x.dtype,
            )
            outputs.append(_conv_same(x[idx:idx + 1], kernel))
        return torch.cat(outputs, dim=0)

    def _repeat_gray_channels(self, x: torch.Tensor) -> torch.Tensor:
        if not self.force_grayscale or x.shape[1] <= 1:
            return x
        gray = x.mean(dim=1, keepdim=True)
        return gray.repeat(1, x.shape[1], 1, 1)

    def _effective_particle_count(self, t: torch.Tensor, flux_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_count = self._base_particle_count(t)
        effective_count = base_count * torch.exp(self.latent_modulation_scale * flux_delta)
        return base_count, effective_count.clamp_min(1.0)

    def apply_source_flux(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        flux_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transmission = clean.clamp(-1.0, 1.0).add(1.0).div(2.0)
        base_count, effective_count = self._effective_particle_count(t, flux_delta)
        return transmission, {
            "transmission": transmission,
            "particle_count_base": base_count,
            "particle_count_effective": effective_count,
            "source_gain": (effective_count / self.flux_ref_max),
        }

    def apply_geometric_blur(self, x: torch.Tensor, blur_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_sigma = self._sigma_geo().to(device=x.device, dtype=x.dtype)
        mod_x = 1.0 + self.blur_modulation_limit * blur_params[:, 0]
        mod_y = 1.0 + self.blur_modulation_limit * blur_params[:, 1]
        sigma_x = (base_sigma * mod_x).clamp_min(1e-4)
        sigma_y = (base_sigma * mod_y).clamp_min(1e-4)
        return self._apply_blur(x, sigma_x, sigma_y, self.geo_kernel_size), sigma_x, sigma_y

    def apply_detector_blur(self, x: torch.Tensor, blur_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_sigma = self._sigma_det().to(device=x.device, dtype=x.dtype)
        mod_x = 1.0 + self.blur_modulation_limit * blur_params[:, 2]
        mod_y = 1.0 + self.blur_modulation_limit * blur_params[:, 3]
        sigma_x = (base_sigma * mod_x).clamp_min(1e-4)
        sigma_y = (base_sigma * mod_y).clamp_min(1e-4)
        return self._apply_blur(x, sigma_x, sigma_y, self.det_kernel_size), sigma_x, sigma_y

    def apply_scatter_background(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        scatter_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = gaussian_kernel2d(
            self.scatter_kernel_size,
            sigma_x=self.scatter_sigma_px,
            sigma_y=self.scatter_sigma_px,
            device=x.device,
            dtype=x.dtype,
        )
        scatter_base = _conv_same(x, kernel)
        particle_ratio = self._base_particle_count(t) / self.particle_count_max
        scarcity = (1.0 - particle_ratio).view(-1, 1, 1, 1)
        strength_map = (self.scatter_strength_max * scarcity * scatter_map).clamp(min=0.0)
        scatter = strength_map * scatter_base
        strength_scalar = strength_map.mean(dim=(1, 2, 3))
        return scatter, strength_scalar

    def apply_poisson_layer(
        self,
        x: torch.Tensor,
        effective_particle_count: torch.Tensor,
        sample_noise: bool = True,
    ) -> torch.Tensor:
        flux = effective_particle_count.view(-1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        expected_counts = torch.clamp(x, min=0.0) * flux
        if self.shared_channel_stochastic and expected_counts.shape[1] > 1:
            expected_gray = expected_counts.mean(dim=1, keepdim=True)
            sampled = torch.poisson(expected_gray) if sample_noise else expected_gray
            sampled = sampled.repeat(1, expected_counts.shape[1], 1, 1)
        else:
            sampled = torch.poisson(expected_counts) if sample_noise else expected_counts
        return sampled / flux.clamp_min(1e-8)

    def apply_readout_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        readout_map: torch.Tensor,
        sample_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        particle_ratio = self._base_particle_count(t) / self.particle_count_max
        scarcity = (1.0 - particle_ratio).view(-1, 1, 1, 1)
        sigma_map = (self.readout_sigma_max * scarcity * readout_map).clamp(min=0.0)
        sigma_scalar = sigma_map.mean(dim=(1, 2, 3))
        if not sample_noise:
            return x, sigma_scalar
        if self.shared_channel_stochastic and x.shape[1] > 1:
            noise = torch.randn(
                x.shape[0],
                1,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            noise = torch.randn_like(x)
        return x + noise * sigma_map, sigma_scalar

    def forward(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None = None,
        sample_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        clean = self._repeat_gray_channels(clean)
        cond = self.condition_encoder(degradation_latent, t, output_hw=clean.shape[-2:])
        source_out, source_stats = self.apply_source_flux(clean, t, cond["flux_delta"])

        geo_out, sigma_geo_x, sigma_geo_y = self.apply_geometric_blur(source_out, cond["blur_params"])
        det_out, sigma_det_x, sigma_det_y = self.apply_detector_blur(geo_out, cond["blur_params"])
        blur_out, interaction_residual = self.blur_interaction(
            geo_out,
            det_out,
            degradation_latent,
            t,
            cond["interaction_gate"],
        )
        blur_out = self._repeat_gray_channels(blur_out)
        interaction_residual = self._repeat_gray_channels(interaction_residual)

        scatter, scatter_strength = self.apply_scatter_background(
            blur_out,
            t,
            cond["scatter_map"],
        )
        pre_poisson = self._repeat_gray_channels(blur_out + scatter)
        poisson_out = self.apply_poisson_layer(
            pre_poisson,
            source_stats["particle_count_effective"],
            sample_noise=sample_noise,
        )
        poisson_out = self._repeat_gray_channels(poisson_out)
        noisy_out, readout_sigma = self.apply_readout_noise(
            poisson_out,
            t,
            cond["readout_map"],
            sample_noise=sample_noise,
        )
        noisy_out = self._repeat_gray_channels(noisy_out)
        degraded = noisy_out.clamp(0.0, 1.0).mul(2.0).sub(1.0)

        return {
            "transmission": source_stats["transmission"],
            "source": source_out,
            "geo_blur": geo_out,
            "det_blur": det_out,
            "blur_interaction": blur_out,
            "interaction_residual": interaction_residual,
            "scatter": scatter,
            "pre_poisson": pre_poisson,
            "degraded": degraded,
            "particle_count_base": source_stats["particle_count_base"],
            "particle_count_effective": source_stats["particle_count_effective"],
            "source_gain": source_stats["source_gain"],
            "sigma_geo_x": sigma_geo_x,
            "sigma_geo_y": sigma_geo_y,
            "sigma_det_x": sigma_det_x,
            "sigma_det_y": sigma_det_y,
            "sigma_geo": 0.5 * (sigma_geo_x + sigma_geo_y),
            "sigma_det": 0.5 * (sigma_det_x + sigma_det_y),
            "scatter_strength": scatter_strength,
            "readout_sigma": readout_sigma,
            "scatter_map": cond["scatter_map"],
            "readout_map": cond["readout_map"],
            "interaction_gate": cond["interaction_gate"],
            "t": t,
        }
