from __future__ import annotations

import torch
import torch.nn.functional as F

from .physics_forward import NeutronPhysicalForward


class GuidedNeutronPhysicalForward(NeutronPhysicalForward):
    def _prepare_override_cond(
        self,
        override_cond: dict[str, torch.Tensor],
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = clean.shape[-2:]
        device = clean.device
        dtype = clean.dtype
        batch = clean.shape[0]

        def _map_value(key: str) -> torch.Tensor:
            value = override_cond[key].to(device=device, dtype=dtype)
            if value.shape[-2:] != (height, width):
                value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
            return value

        blur_params = override_cond["blur_params"].to(device=device, dtype=dtype)
        if blur_params.ndim == 1:
            blur_params = blur_params.unsqueeze(0)
        flux_delta = override_cond["flux_delta"].to(device=device, dtype=dtype).view(batch)
        return {
            "blur_params": blur_params,
            "flux_delta": flux_delta,
            "scatter_map": _map_value("scatter_map"),
            "readout_map": _map_value("readout_map"),
            "interaction_gate": _map_value("interaction_gate"),
        }

    def forward(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        degradation_latent: torch.Tensor | None = None,
        sample_noise: bool = True,
        override_cond: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if override_cond is None:
            cond = self.condition_encoder(degradation_latent, t, output_hw=clean.shape[-2:])
        else:
            cond = self._prepare_override_cond(override_cond, clean=clean, t=t)

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

        scatter, scatter_strength = self.apply_scatter_background(
            blur_out,
            t,
            cond["scatter_map"],
        )
        pre_poisson = blur_out + scatter
        poisson_out = self.apply_poisson_layer(
            pre_poisson,
            source_stats["particle_count_effective"],
            sample_noise=sample_noise,
        )
        noisy_out, readout_sigma = self.apply_readout_noise(
            poisson_out,
            t,
            cond["readout_map"],
            sample_noise=sample_noise,
        )
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
            "flux_delta": cond["flux_delta"],
            "blur_params": cond["blur_params"],
            "t": t,
        }
