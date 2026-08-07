from __future__ import annotations

from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torchmetrics import MeanMetric

from utils.common import frozen_module, instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt
from utils.neutron_schedule import LearnableTParameters


LEVEL_KEYS = ["LQ_50", "LQ_30", "LQ_20", "LQ_10"]


class LitGuidedDegradation(pl.LightningModule):
    def __init__(
        self,
        misc_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        predictor_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        physics_config: Mapping[str, Any],
        guided_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.guided_config = guided_config

        self.autoencoder = instantiate_from_config(ae_config)
        self.predictor = instantiate_from_config(predictor_config)
        self.physics_forward = instantiate_from_config(physics_config)

        self.ae_ckpt_path = ae_config["checkpoint"]
        self.t_learnable = guided_config.get("t_learnable", False)
        self.t_params = LearnableTParameters(learnable=self.t_learnable)

        self.loss_weights = {
            "img": float(guided_config.get("img_loss_scale", 1.0)),
            "z": float(guided_config.get("z_loss_scale", 0.5)),
            "mono": float(guided_config.get("monotonic_loss_scale", 0.1)),
            "smooth": float(guided_config.get("smooth_loss_scale", 0.05)),
            "tv": float(guided_config.get("tv_loss_scale", 0.01)),
        }
        self.charbonnier_eps = float(guided_config.get("charbonnier_eps", 1e-3))
        self.physics_finetune_start = int(guided_config.get("physics_finetune_start", -1))
        self.physics_lr_scale = float(guided_config.get("physics_lr_scale", 0.25))
        self.physics_trainable_modules = list(guided_config.get("physics_trainable_modules", ["condition_encoder", "blur_interaction"]))
        self.physics_trainable_scalars = list(guided_config.get("physics_trainable_scalars", []))
        self.test_y_channel = bool(guided_config.get("test_y_channel", True))

        self.physics_finetune_enabled = False
        self.val_psnr = nn.ModuleDict({key: MeanMetric(dist_sync_on_step=True) for key in LEVEL_KEYS})

    def setup(self, stage: Optional[str] = None) -> None:
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.logger is not None and hasattr(trainer.logger, "version"):
            self.version = trainer.logger.version
        else:
            self.version = "temp"

        checkpoint = torch.load(self.ae_ckpt_path, map_location="cpu")
        required_keys = [
            "content_encoder",
            "global_degradation_encoder",
            "local_degradation_encoder",
            "degradation_compressor",
            "decoder",
            "t_params",
        ]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(
                "The guided stage-2 model requires a rewritten stage-1 checkpoint. "
                f"Missing keys in AE checkpoint: {missing_keys}"
            )

        self.autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
        self.autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
        self.autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
        self.autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
        self.autoencoder.decoder.load_state_dict(checkpoint["decoder"])
        self.t_params.load_export_state_dict(checkpoint["t_params"])
        if "physics_forward" in checkpoint:
            self.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)

        frozen_module(self.autoencoder.content_encoder)
        frozen_module(self.autoencoder.global_degradation_encoder)
        frozen_module(self.autoencoder.local_degradation_encoder)
        frozen_module(self.autoencoder.degradation_compressor)
        frozen_module(self.autoencoder.decoder)

        for param in self.physics_forward.parameters():
            param.requires_grad = False
        for scalar_name in ("log_sigma_geo", "log_sigma_det"):
            if hasattr(self.physics_forward, scalar_name):
                getattr(self.physics_forward, scalar_name).requires_grad = False

        if self.physics_finetune_start == 0:
            self._enable_physics_finetune()

    def configure_optimizers(self):
        optim_groups = [{"params": list(self.predictor.parameters())}]

        physics_params = list(self.physics_forward.parameters())
        if physics_params:
            physics_group = {"params": physics_params}
            base_lr = self.optimizer_config.get("params", {}).get("lr")
            if base_lr is not None:
                physics_group["lr"] = float(base_lr) * self.physics_lr_scale
            optim_groups.append(physics_group)

        if self.t_learnable:
            optim_groups.append({"params": list(self.t_params.parameters())})

        optimizer = instantiate_from_config_with_arg(self.optimizer_config, optim_groups)
        if self.scheduler_config is None:
            return {"optimizer": optimizer}

        scheduler = instantiate_from_config_with_arg(self.scheduler_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    def on_train_batch_start(self, batch, batch_idx):
        x = batch["GT"]
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()
        if (
            not self.physics_finetune_enabled
            and self.physics_finetune_start >= 0
            and self.trainer is not None
            and self.trainer.global_step >= self.physics_finetune_start
        ):
            self._enable_physics_finetune()

    def on_before_optimizer_step(self, optimizer):
        warmup_iter = self.misc_config.get("warmup")
        if warmup_iter is None or warmup_iter <= 0 or self.trainer is None:
            return
        if self.trainer.global_step >= warmup_iter:
            return

        lr_scale = min(1.0, float(self.trainer.global_step + 1) / warmup_iter)
        base_lr = float(self.optimizer_config.params.lr)
        for param_group in optimizer.param_groups:
            group_lr = float(param_group.get("lr", base_lr))
            scaled_base = group_lr if self.trainer.global_step == 0 else float(param_group.get("_base_lr", group_lr))
            param_group["_base_lr"] = scaled_base
            param_group["lr"] = lr_scale * scaled_base

    def _enable_physics_finetune(self) -> None:
        for module_name in self.physics_trainable_modules:
            if hasattr(self.physics_forward, module_name):
                for param in getattr(self.physics_forward, module_name).parameters():
                    param.requires_grad = True
        for scalar_name in self.physics_trainable_scalars:
            if hasattr(self.physics_forward, scalar_name):
                getattr(self.physics_forward, scalar_name).requires_grad = True
        self.physics_finetune_enabled = True

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.charbonnier_eps ** 2).mean()

    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        loss_h = torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :]))
        loss_w = torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
        return loss_h + loss_w

    def monotonic_loss(self, outputs_by_level: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        increasing_metrics = ["sigma_geo", "sigma_det", "scatter_strength", "readout_sigma"]
        decreasing_metrics = ["particle_count_effective"]
        total = torch.zeros((), device=self.device)

        for metric in increasing_metrics:
            stacked = torch.stack([outputs_by_level[key][metric].view(-1) for key in LEVEL_KEYS], dim=0)
            total = total + F.relu(stacked[:-1] - stacked[1:]).mean()

        for metric in decreasing_metrics:
            stacked = torch.stack([outputs_by_level[key][metric].view(-1) for key in LEVEL_KEYS], dim=0)
            total = total + F.relu(stacked[1:] - stacked[:-1]).mean()

        return total

    def smoothness_loss(
        self,
        predictions_by_level: dict[str, dict[str, torch.Tensor]],
        outputs_by_level: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        z_stack = torch.stack([predictions_by_level[key]["pred_z"] for key in LEVEL_KEYS], dim=0)
        if z_stack.shape[0] < 3:
            return torch.zeros((), device=z_stack.device)

        total = torch.mean(torch.abs(z_stack[:-2] - 2.0 * z_stack[1:-1] + z_stack[2:]))

        blur_stack = torch.stack([predictions_by_level[key]["blur_params"] for key in LEVEL_KEYS], dim=0)
        flux_stack = torch.stack([predictions_by_level[key]["flux_delta"].view(-1, 1) for key in LEVEL_KEYS], dim=0)
        total = total + torch.mean(torch.abs(blur_stack[:-2] - 2.0 * blur_stack[1:-1] + blur_stack[2:]))
        total = total + torch.mean(torch.abs(flux_stack[:-2] - 2.0 * flux_stack[1:-1] + flux_stack[2:]))

        readout_stack = torch.stack([outputs_by_level[key]["readout_sigma"].view(-1, 1) for key in LEVEL_KEYS], dim=0)
        total = total + torch.mean(torch.abs(readout_stack[:-2] - 2.0 * readout_stack[1:-1] + readout_stack[2:]))
        return total

    def _level_t(self, key: str, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_value = self.t_params.as_dict(device=device, dtype=dtype)[key]
        return t_value.expand(batch_size)

    def _predict_level(
        self,
        clean: torch.Tensor,
        z_content: torch.Tensor,
        key: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        t = self._level_t(key, batch_size=clean.shape[0], device=clean.device, dtype=clean.dtype)
        prediction = self.predictor(z_content=z_content, clean=clean, t=t)
        physics_out = self.physics_forward(
            clean=clean,
            t=t,
            degradation_latent=prediction["pred_z"],
            sample_noise=False,
            override_cond=prediction,
        )
        return prediction, physics_out

    def _shared_step(self, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        clean = batch["GT"]
        content_dict = self.autoencoder.encode_content(clean)
        z_content = content_dict["z_c"]

        predictions_by_level: dict[str, dict[str, torch.Tensor]] = {}
        outputs_by_level: dict[str, dict[str, torch.Tensor]] = {}
        targets_by_level: dict[str, torch.Tensor] = {}

        loss_img = torch.zeros((), device=clean.device)
        loss_z = torch.zeros((), device=clean.device)
        loss_tv = torch.zeros((), device=clean.device)

        for key in LEVEL_KEYS:
            with torch.no_grad():
                target_z = self.autoencoder.encode_degradation(batch[key])["z_t"]
            prediction, physics_out = self._predict_level(clean=clean, z_content=z_content, key=key)

            predictions_by_level[key] = prediction
            outputs_by_level[key] = physics_out
            targets_by_level[key] = batch[key]

            loss_img = loss_img + self.charbonnier_loss(physics_out["degraded"], batch[key])
            loss_z = loss_z + F.l1_loss(prediction["pred_z"], target_z)
            loss_tv = loss_tv + self.total_variation_loss(prediction["scatter_map"])
            loss_tv = loss_tv + self.total_variation_loss(prediction["readout_map"])

        level_count = float(len(LEVEL_KEYS))
        loss_img = loss_img / level_count
        loss_z = loss_z / level_count
        loss_tv = loss_tv / (2.0 * level_count)
        loss_mono = self.monotonic_loss(outputs_by_level)
        loss_smooth = self.smoothness_loss(predictions_by_level, outputs_by_level)

        total_loss = (
            self.loss_weights["img"] * loss_img
            + self.loss_weights["z"] * loss_z
            + self.loss_weights["mono"] * loss_mono
            + self.loss_weights["smooth"] * loss_smooth
            + self.loss_weights["tv"] * loss_tv
        )

        log_prefix = "train" if mode == "train" else "val"
        self.log_dict(
            {
                f"{log_prefix}/loss": total_loss,
                f"{log_prefix}/img": loss_img,
                f"{log_prefix}/z": loss_z,
                f"{log_prefix}/mono": loss_mono,
                f"{log_prefix}/smooth": loss_smooth,
                f"{log_prefix}/tv": loss_tv,
                f"{log_prefix}/sigma_geo_10": outputs_by_level["LQ_10"]["sigma_geo"].mean(),
                f"{log_prefix}/sigma_det_10": outputs_by_level["LQ_10"]["sigma_det"].mean(),
                f"{log_prefix}/readout_10": outputs_by_level["LQ_10"]["readout_sigma"].mean(),
            },
            prog_bar=(mode == "train"),
            on_step=(mode == "train"),
            on_epoch=True,
            sync_dist=(mode != "train"),
        )

        if mode == "val":
            for key in LEVEL_KEYS:
                pred = outputs_by_level[key]["degraded"].clamp(-1, 1).add(1).div(2)
                target = targets_by_level[key].add(1).div(2)
                psnr_val = calculate_psnr_pt(target, pred, crop_border=0, test_y_channel=self.test_y_channel)
                self.val_psnr[key].update(psnr_val)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, mode="val")

    def on_validation_epoch_end(self):
        logs = {}
        for key, metric in self.val_psnr.items():
            logs[f"psnr/{key}"] = metric.compute()
            metric.reset()
        self.log_dict(logs, prog_bar=True, sync_dist=True, rank_zero_only=True, on_epoch=True)
