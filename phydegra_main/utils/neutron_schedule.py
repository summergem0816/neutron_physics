import math
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


PARTICLE_COUNT_BY_KEY = {
    "GT": 300.0,
    "LQ_50": 50.0,
    "LQ_30": 30.0,
    "LQ_20": 20.0,
    "LQ_10": 10.0,
}

TRAJECTORY_ORDER = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]
LEARNABLE_T_KEYS = TRAJECTORY_ORDER[1:]


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


def export_t_map_from_state_dict(
    state_dict: Mapping[str, torch.Tensor | float] | None,
    device=None,
    dtype=torch.float32,
) -> dict[str, torch.Tensor]:
    schedule = LearnableTParameters(learnable=False)
    if state_dict is not None:
        schedule.load_export_state_dict(state_dict)
    return schedule.as_dict(device=device, dtype=dtype)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x).clamp_min(1e-8))


class LearnableTParameters(nn.Module):
    def __init__(
        self,
        learnable: bool,
        init_t_map: Mapping[str, float] | None = None,
        gap_floor: float = 1e-3,
    ):
        super().__init__()
        self.learnable = learnable
        self.gap_floor = float(gap_floor)
        init_t_map = dict(build_t_map() if init_t_map is None else init_t_map)
        init_gaps = self._map_to_gaps(init_t_map)

        self.raw_gaps = nn.ParameterDict()
        for key in LEARNABLE_T_KEYS:
            gap = max(float(init_gaps[key]), self.gap_floor * 2.0)
            raw = _inverse_softplus(torch.tensor(gap - self.gap_floor, dtype=torch.float32))
            self.raw_gaps[key] = nn.Parameter(raw, requires_grad=learnable)

    def _map_to_gaps(self, t_map: Mapping[str, float]) -> dict[str, float]:
        prev = 0.0
        gaps = {}
        for key in LEARNABLE_T_KEYS:
            cur = float(t_map[key])
            gaps[key] = max(cur - prev, self.gap_floor)
            prev = cur
        return gaps

    def _gap_tensors(self, device=None, dtype=None) -> list[torch.Tensor]:
        gaps = []
        for key in LEARNABLE_T_KEYS:
            gap = F.softplus(self.raw_gaps[key]) + self.gap_floor
            if device is not None or dtype is not None:
                gap = gap.to(device=device or gap.device, dtype=dtype or gap.dtype)
            gaps.append(gap)
        return gaps

    def positions_tensor(self, device=None, dtype=None) -> torch.Tensor:
        gaps = self._gap_tensors(device=device, dtype=dtype)
        gap_tensor = torch.stack(gaps, dim=0)
        normalized_gaps = gap_tensor / gap_tensor.sum().clamp_min(1e-8)
        cumulative = torch.cumsum(normalized_gaps, dim=0)
        t_gt = torch.zeros(1, device=cumulative.device, dtype=cumulative.dtype)
        return torch.cat([t_gt, cumulative], dim=0)

    def as_dict(self, device=None, dtype=None) -> dict[str, torch.Tensor]:
        positions = self.positions_tensor(device=device, dtype=dtype)
        return {
            key: positions[idx]
            for idx, key in enumerate(TRAJECTORY_ORDER)
        }

    def export_state_dict(self, device=None, dtype=torch.float32) -> dict[str, torch.Tensor]:
        t_map = self.as_dict(device=device, dtype=dtype)
        return {
            key: value.detach().clone()
            for key, value in t_map.items()
            if key != "GT"
        }

    def load_export_state_dict(self, state_dict: Mapping[str, torch.Tensor | float]) -> None:
        if any(key.startswith("raw_gaps.") for key in state_dict.keys()):
            self.load_state_dict(state_dict, strict=True)
            return

        init_t_map = {"GT": 0.0}
        for key in LEARNABLE_T_KEYS:
            if key not in state_dict:
                raise KeyError(f"Missing {key} in t_params state dict")
            value = state_dict[key]
            if torch.is_tensor(value):
                value = float(value.detach().cpu().item())
            init_t_map[key] = float(value)

        init_gaps = self._map_to_gaps(init_t_map)
        with torch.no_grad():
            for key in LEARNABLE_T_KEYS:
                gap = max(float(init_gaps[key]), self.gap_floor * 2.0)
                raw = _inverse_softplus(torch.tensor(gap - self.gap_floor, dtype=self.raw_gaps[key].dtype))
                self.raw_gaps[key].copy_(raw.to(device=self.raw_gaps[key].device))

    def batch_values(self, keys: Iterable[str], device=None, dtype=None) -> torch.Tensor:
        t_map = self.as_dict(device=device, dtype=dtype)
        values = []
        for key in keys:
            values.append(t_map[str(key)])
        return torch.stack(values, dim=0)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.as_dict()[key]
