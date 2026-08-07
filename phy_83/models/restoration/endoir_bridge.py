from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict at {path}, got {type(data)!r}")
    return data


class FrozenEndoIR(nn.Module):
    """Frozen EndoIR inference wrapper used as the restoration front-end."""

    def __init__(
        self,
        repo_root: str,
        config_path: str,
        weight_path: str,
        tone_weight_path: str | None = None,
        strict_load_g: bool = False,
        strict_load_tone: bool = True,
        ddim_timesteps: int | None = None,
        ddim_eta: float | None = None,
        pyramid_list: list[int] | None = None,
        force_gray_output: bool | None = None,
        gray_method: str | None = None,
        gray_scale: float | None = None,
        gray_shift: float | None = None,
        pad_divide: int = 32,
    ):
        super().__init__()

        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.weight_path = Path(weight_path).resolve()
        self.tone_weight_path = Path(tone_weight_path).resolve() if tone_weight_path else None
        self.pad_divide = int(pad_divide)

        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"EndoIR repo root not found: {self.repo_root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"EndoIR config not found: {self.config_path}")
        if not self.weight_path.is_file():
            raise FileNotFoundError(
                "EndoIR weight file not found. "
                f"Expected at: {self.weight_path}"
            )
        if self.tone_weight_path is not None and not self.tone_weight_path.is_file():
            raise FileNotFoundError(f"EndoIR tone weight file not found: {self.tone_weight_path}")

        self._repo_sys_path = str(self.repo_root)
        if self._repo_sys_path not in sys.path:
            sys.path.insert(0, self._repo_sys_path)

        importlib.import_module("endoir.archs")
        importlib.import_module("endoir.models")

        model_module = importlib.import_module("endoir.models.EndoIR_model")
        model_cls = getattr(model_module, "EndoIR")

        opt = _load_yaml(self.config_path)
        opt["is_train"] = False
        opt["dist"] = False
        opt["num_gpu"] = 1
        opt["rank"] = 0

        path_cfg = dict(opt.get("path", {}))
        path_cfg["pretrain_network_g"] = str(self.weight_path)
        path_cfg["strict_load_g"] = bool(strict_load_g)
        if self.tone_weight_path is not None:
            path_cfg["pretrain_network_tone"] = str(self.tone_weight_path)
            path_cfg["strict_load_tone"] = bool(strict_load_tone)
        else:
            path_cfg["pretrain_network_tone"] = None
        opt["path"] = path_cfg

        val_cfg = dict(opt.get("val", {}))
        if ddim_timesteps is not None:
            val_cfg["ddim_timesteps"] = int(ddim_timesteps)
        if ddim_eta is not None:
            val_cfg["ddim_eta"] = float(ddim_eta)
        if pyramid_list is not None:
            val_cfg["pyramid_list"] = list(pyramid_list)
        if force_gray_output is not None:
            val_cfg["force_gray_output"] = bool(force_gray_output)
        if gray_method is not None:
            val_cfg["gray_method"] = str(gray_method)
        if gray_scale is not None:
            val_cfg["gray_scale"] = float(gray_scale)
        if gray_shift is not None:
            val_cfg["gray_shift"] = float(gray_shift)
        opt["val"] = val_cfg

        self.model = model_cls(opt)
        self.model.ddpm.eval()
        if self.model.tone_corrector is not None:
            self.model.tone_corrector.eval()

        for param in self.model.ddpm.parameters():
            param.requires_grad = False
        if self.model.tone_corrector is not None:
            for param in self.model.tone_corrector.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(False)
        self.model.ddpm.eval()
        if self.model.tone_corrector is not None:
            self.model.tone_corrector.eval()
        return self

    def _pad_if_needed(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        height, width = x.shape[-2:]
        pad_h = (self.pad_divide - height % self.pad_divide) % self.pad_divide
        pad_w = (self.pad_divide - width % self.pad_divide) % self.pad_divide
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        return padded, (pad_left, pad_right, pad_top, pad_bottom)

    @staticmethod
    def _unpad(x: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
        pad_left, pad_right, pad_top, pad_bottom = pads
        if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
            return x
        return x[..., pad_top:x.shape[-2] - pad_bottom, pad_left:x.shape[-1] - pad_right]

    @torch.no_grad()
    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        lq_padded, pads = self._pad_if_needed(lq)
        fake_hq = lq_padded

        self.model.LR = lq_padded
        self.model.HR = fake_hq
        self.model.test()

        restored = self.model.output
        restored = self._unpad(restored, pads)
        return restored.clamp(-1.0, 1.0)
