import math

import torch


PARTICLE_COUNT_BY_KEY = {
    "GT": 300.0,
    "LQ_50": 50.0,
    "LQ_30": 30.0,
    "LQ_20": 20.0,
    "LQ_10": 10.0,
}

TRAJECTORY_ORDER = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]


def particle_count_to_t(
    particle_count,
    particle_count_max: float = PARTICLE_COUNT_BY_KEY["GT"],
    particle_count_min: float = PARTICLE_COUNT_BY_KEY["LQ_10"],
):
    log_max = math.log(particle_count_max)
    log_min = math.log(particle_count_min)
    denom = max(log_max - log_min, 1e-8)

    if isinstance(particle_count, torch.Tensor):
        value = particle_count.to(dtype=torch.float32).clamp(min=particle_count_min, max=particle_count_max)
        t = (math.log(particle_count_max) - torch.log(value)) / denom
        return torch.clamp(t, 0.0, 1.0)

    value = float(particle_count)
    value = min(max(value, particle_count_min), particle_count_max)
    t = (log_max - math.log(value)) / denom
    return float(min(max(t, 0.0), 1.0))


def build_t_map() -> dict[str, float]:
    return {
        key: particle_count_to_t(count)
        for key, count in PARTICLE_COUNT_BY_KEY.items()
    }


def build_t_tensor_dict(device=None, dtype=torch.float32) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, device=device, dtype=dtype)
        for key, value in build_t_map().items()
    }
