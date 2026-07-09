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
from utils.neutron_schedule import LEARNABLE_T_KEYS, LearnableTParameters


LEVEL_KEYS = ["LQ_50", "LQ_30", "LQ_20", "LQ_10"]


class LitZOnlyDegradation(pl.LightningModule):
    def __init__(
        self,
        misc_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        predictor_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        physics_config: Mapping[str, Any],
        zonly_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.zonly_config = zonly_config

        self.autoencoder = instantiate_from_config(ae_config)
        self.predictor = instantiate_from_config(predictor_config)
        self.physics_forward = instantiate_from_config(physics_config) if physics_config else None

        self.ae_ckpt_path = ae_config["checkpoint"]
        self.t_learnable = bool(zonly_config.get("t_learnable", False))
        self.t_params = LearnableTParameters(learnable=self.t_learnable)

        self.img_loss_scale = float(zonly_config.get("img_loss_scale", 1.0))
        self.z_loss_scale = float(zonly_config.get("z_loss_scale", 0.5))
        self.z_cosine_weight = float(zonly_config.get("z_cosine_weight", 0.2))
        self.t_anchor_reg_scale = float(zonly_config.get("t_anchor_reg_scale", 0.0))
        self.charbonnier_eps = float(zonly_config.get("charbonnier_eps", 1e-3))
        self.test_y_channel = bool(zonly_config.get("test_y_channel", True))

        self.val_psnr = nn.ModuleDict({key: MeanMetric(dist_sync_on_step=True) for key in LEVEL_KEYS})
        self.register_buffer(
            "t_anchor_positions",
            torch.zeros(len(LEVEL_KEYS) + 1, dtype=torch.float32),
            persistent=False,
        )

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
                "The z-only stage-2 model requires a rewritten stage-1 checkpoint. "
                f"Missing keys in AE checkpoint: {missing_keys}"
            )

        self.autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
        self.autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
        self.autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
        self.autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
        self.autoencoder.decoder.load_state_dict(checkpoint["decoder"])
        self.t_params.load_export_state_dict(checkpoint["t_params"])
        with torch.no_grad():
            self.t_anchor_positions.copy_(
                self.t_params.positions_tensor(device=self.device, dtype=torch.float32).detach().to(self.t_anchor_positions)
            )

        if self.physics_forward is not None and "physics_forward" in checkpoint:
            self.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)

        frozen_module(self.autoencoder.content_encoder)
        frozen_module(self.autoencoder.global_degradation_encoder)
        frozen_module(self.autoencoder.local_degradation_encoder)
        frozen_module(self.autoencoder.degradation_compressor)
        frozen_module(self.autoencoder.decoder)
        if self.physics_forward is not None:
            frozen_module(self.physics_forward)

    def configure_optimizers(self):
        optim_groups = [{"params": list(self.predictor.parameters())}]
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

    def on_before_optimizer_step(self, optimizer):
        warmup_iter = self.misc_config.get("warmup")
        trainer = getattr(self, "_trainer", None)
        if warmup_iter is None or warmup_iter <= 0 or trainer is None:
            return
        if trainer.global_step >= warmup_iter:
            return

        lr_scale = min(1.0, float(trainer.global_step + 1) / warmup_iter)
        base_lr = float(self.optimizer_config.params.lr)
        for param_group in optimizer.param_groups:
            group_lr = float(param_group.get("lr", base_lr))
            scaled_base = group_lr if trainer.global_step == 0 else float(param_group.get("_base_lr", group_lr))
            param_group["_base_lr"] = scaled_base
            param_group["lr"] = lr_scale * scaled_base

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.charbonnier_eps ** 2).mean()

    def latent_alignment_loss(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_z.flatten(1)
        target_flat = target_z.flatten(1)
        cosine_penalty = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1)
        return F.l1_loss(pred_z, target_z) + self.z_cosine_weight * cosine_penalty.mean()

    def t_anchor_loss(self) -> torch.Tensor:
        current = self.t_params.positions_tensor(device=self.device, dtype=torch.float32)
        anchor = self.t_anchor_positions.to(device=current.device, dtype=current.dtype)
        return F.mse_loss(current[1:], anchor[1:], reduction="mean")

    def _level_t(self, key: str, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_value = self.t_params.as_dict(device=device, dtype=dtype)[key]
        return t_value.expand(batch_size)

    def _shared_step(self, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        clean = batch["GT"]
        content_dict = self.autoencoder.encode_content(clean)
        z_content = content_dict["z_c"]

        loss_img = torch.zeros((), device=clean.device)
        loss_z = torch.zeros((), device=clean.device)
        outputs_by_level: dict[str, dict[str, torch.Tensor]] = {}

        for key in LEVEL_KEYS:
            with torch.no_grad():
                target_z = self.autoencoder.encode_degradation(batch[key])["z_t"]

            t = self._level_t(key, batch_size=clean.shape[0], device=clean.device, dtype=clean.dtype)
            pred_z = self.predictor(z_content=z_content, clean=clean, t=t)["pred_z"]
            if self.physics_forward is not None:
                physics_out = self.physics_forward(
                    clean=clean,
                    t=t,
                    degradation_latent=pred_z,
                    sample_noise=False,
                )
                pred_img = physics_out["degraded"]
                outputs_by_level[key] = physics_out
            else:
                pred_img = self.autoencoder.reconstruct_from_content_degradation(pred_z, content_dict["features"])

            loss_img = loss_img + self.charbonnier_loss(pred_img, batch[key])
            loss_z = loss_z + self.latent_alignment_loss(pred_z, target_z)

        level_count = float(len(LEVEL_KEYS))
        loss_img = loss_img / level_count
        loss_z = loss_z / level_count
        total_loss = self.img_loss_scale * loss_img + self.z_loss_scale * loss_z

        if self.t_learnable and self.t_anchor_reg_scale > 0:
            loss_t_anchor = self.t_anchor_loss()
            total_loss = total_loss + self.t_anchor_reg_scale * loss_t_anchor
        else:
            loss_t_anchor = torch.zeros((), device=clean.device)

        log_prefix = "train" if mode == "train" else "val"
        log_dict = {
            f"{log_prefix}/loss": total_loss,
            f"{log_prefix}/img": loss_img,
            f"{log_prefix}/z": loss_z,
            f"{log_prefix}/t_anchor": loss_t_anchor,
        }
        for key in LEARNABLE_T_KEYS:
            log_dict[f"{log_prefix}/t_{key.split('_')[1].lower()}"] = self.t_params.as_dict(
                device=clean.device,
                dtype=clean.dtype,
            )[key]
        if self.physics_forward is not None and "LQ_10" in outputs_by_level:
            log_dict[f"{log_prefix}/sigma_geo_10"] = outputs_by_level["LQ_10"]["sigma_geo"].mean()
            log_dict[f"{log_prefix}/sigma_det_10"] = outputs_by_level["LQ_10"]["sigma_det"].mean()
            log_dict[f"{log_prefix}/readout_10"] = outputs_by_level["LQ_10"]["readout_sigma"].mean()

        self.log_dict(
            log_dict,
            prog_bar=(mode == "train"),
            on_step=(mode == "train"),
            on_epoch=True,
            sync_dist=(mode != "train"),
        )

        if mode == "val":
            for key in LEVEL_KEYS:
                pred = outputs_by_level[key]["degraded"].clamp(-1, 1).add(1).div(2)
                target = batch[key].add(1).div(2)
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
