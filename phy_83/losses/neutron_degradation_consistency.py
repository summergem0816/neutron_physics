from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _odd_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 3:
        return 3
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    kernel_size = _odd_kernel_size(kernel_size)
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / max(2.0 * sigma * sigma, 1e-8))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_lowpass(x: torch.Tensor, kernel_size: int = 21, sigma: float = 2.0) -> torch.Tensor:
    kernel_size = _odd_kernel_size(kernel_size)
    kernel = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=x.shape[1])


def sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.repeat(x.shape[1], 1, 1, 1)
    kernel_y = kernel_y.repeat(x.shape[1], 1, 1, 1)
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(padded, kernel_x, groups=x.shape[1])
    grad_y = F.conv2d(padded, kernel_y, groups=x.shape[1])
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)


def laplacian_energy(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, groups=x.shape[1])
    return lap.square().mean(dim=(1, 2, 3))


def local_variance_map(residual: torch.Tensor, window_size: int = 9) -> torch.Tensor:
    window_size = _odd_kernel_size(window_size)
    pad = window_size // 2
    mean = F.avg_pool2d(F.pad(residual, (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
    mean_sq = F.avg_pool2d(F.pad(residual.square(), (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
    return (mean_sq - mean.square()).clamp_min(0.0)


def noise_statistics(
    x: torch.Tensor,
    lowpass_kernel_size: int = 21,
    lowpass_sigma: float = 2.0,
    local_window_size: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = gaussian_lowpass(x, kernel_size=lowpass_kernel_size, sigma=lowpass_sigma)
    residual = x - low
    local_var = local_variance_map(residual, window_size=local_window_size)
    grad = sobel_gradient(residual)

    stats = torch.stack(
        [
            residual.mean(dim=(1, 2, 3)),
            residual.abs().mean(dim=(1, 2, 3)),
            residual.flatten(1).std(dim=1, unbiased=False),
            local_var.mean(dim=(1, 2, 3)),
            local_var.flatten(1).std(dim=1, unbiased=False),
            grad.mean(dim=(1, 2, 3)),
            laplacian_energy(residual),
        ],
        dim=1,
    )
    return stats, local_var


class DegradationConsistencyLoss(nn.Module):
    def __init__(
        self,
        mean_loss_scale: float = 1.0,
        structure_loss_scale: float = 0.0,
        noise_stats_loss_scale: float = 0.0,
        noise_map_loss_scale: float = 0.0,
        z_loss_scale: float = 0.5,
        z_highfreq_loss_scale: float = 0.0,
        z_cosine_weight: float = 0.2,
        charbonnier_eps: float = 1e-3,
        lowpass_kernel_size: int = 21,
        lowpass_sigma: float = 2.0,
        noise_local_window_size: int = 9,
        noise_map_pool_size: int = 8,
    ):
        super().__init__()
        self.mean_loss_scale = float(mean_loss_scale)
        self.structure_loss_scale = float(structure_loss_scale)
        self.noise_stats_loss_scale = float(noise_stats_loss_scale)
        self.noise_map_loss_scale = float(noise_map_loss_scale)
        self.z_loss_scale = float(z_loss_scale)
        self.z_highfreq_loss_scale = float(z_highfreq_loss_scale)
        self.z_cosine_weight = float(z_cosine_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.lowpass_kernel_size = int(lowpass_kernel_size)
        self.lowpass_sigma = float(lowpass_sigma)
        self.noise_local_window_size = int(noise_local_window_size)
        self.noise_map_pool_size = int(noise_map_pool_size)

    @property
    def needs_noise_sample(self) -> bool:
        return self.noise_stats_loss_scale > 0 or self.noise_map_loss_scale > 0

    def charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).square() + self.charbonnier_eps**2).mean()

    def latent_alignment_loss(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_z.flatten(1)
        target_flat = target_z.flatten(1)
        cosine_penalty = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1)
        return F.l1_loss(pred_z, target_z) + self.z_cosine_weight * cosine_penalty.mean()

    def latent_highfreq_loss(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        pred_high = pred_z - gaussian_lowpass(pred_z, self.lowpass_kernel_size, self.lowpass_sigma)
        target_high = target_z - gaussian_lowpass(target_z, self.lowpass_kernel_size, self.lowpass_sigma)
        return F.l1_loss(pred_high, target_high.detach())

    def _pooled_noise_map(self, local_var: torch.Tensor) -> torch.Tensor:
        if self.noise_map_pool_size <= 1:
            return local_var
        return F.avg_pool2d(local_var, kernel_size=self.noise_map_pool_size, stride=self.noise_map_pool_size)

    def forward(
        self,
        pred_mean: torch.Tensor,
        real_lq: torch.Tensor,
        pred_z: torch.Tensor | None = None,
        target_z: torch.Tensor | None = None,
        pred_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = pred_mean.new_zeros(())
        parts: dict[str, torch.Tensor] = {
            "mean": zero,
            "structure": zero,
            "noise_stats": zero,
            "noise_map": zero,
            "z": zero,
            "z_highfreq": zero,
        }

        pred_low = gaussian_lowpass(pred_mean, self.lowpass_kernel_size, self.lowpass_sigma)
        real_low = gaussian_lowpass(real_lq, self.lowpass_kernel_size, self.lowpass_sigma)
        parts["mean"] = self.charbonnier(pred_low, real_low)

        if self.structure_loss_scale > 0:
            parts["structure"] = self.charbonnier(sobel_gradient(pred_low), sobel_gradient(real_low))

        if pred_z is not None and target_z is not None:
            parts["z"] = self.latent_alignment_loss(pred_z, target_z)
            if self.z_highfreq_loss_scale > 0:
                parts["z_highfreq"] = self.latent_highfreq_loss(pred_z, target_z)

        if self.needs_noise_sample and pred_sample is not None:
            pred_stats, pred_local_var = noise_statistics(
                pred_sample,
                self.lowpass_kernel_size,
                self.lowpass_sigma,
                self.noise_local_window_size,
            )
            real_stats, real_local_var = noise_statistics(
                real_lq,
                self.lowpass_kernel_size,
                self.lowpass_sigma,
                self.noise_local_window_size,
            )
            parts["noise_stats"] = F.l1_loss(pred_stats, real_stats.detach())
            parts["noise_map"] = F.l1_loss(
                self._pooled_noise_map(pred_local_var),
                self._pooled_noise_map(real_local_var).detach(),
            )

        total = (
            self.mean_loss_scale * parts["mean"]
            + self.structure_loss_scale * parts["structure"]
            + self.noise_stats_loss_scale * parts["noise_stats"]
            + self.noise_map_loss_scale * parts["noise_map"]
            + self.z_loss_scale * parts["z"]
            + self.z_highfreq_loss_scale * parts["z_highfreq"]
        )
        parts["total"] = total
        return total, parts
