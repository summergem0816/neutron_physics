import torch
import torch.nn.functional as F


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def poisson_consistency_loss(pred_counts: torch.Tensor, target_counts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_counts = pred_counts.clamp_min(eps)
    target_counts = target_counts.clamp_min(0.0)
    return torch.mean(pred_counts - target_counts * torch.log(pred_counts))


def scatter_smoothness_loss(scatter_map: torch.Tensor) -> torch.Tensor:
    dy = scatter_map[:, :, 1:, :] - scatter_map[:, :, :-1, :]
    dx = scatter_map[:, :, :, 1:] - scatter_map[:, :, :, :-1]
    return dy.abs().mean() + dx.abs().mean()
