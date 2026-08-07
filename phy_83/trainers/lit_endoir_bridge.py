from __future__ import annotations

from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import nn
from torchmetrics import MeanMetric

from losses.neutron_degradation_consistency import DegradationConsistencyLoss
from utils.common import frozen_module, instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt
from utils.neutron_schedule import LEARNABLE_T_KEYS, LearnableTParameters


class LitEndoIRBridge(pl.LightningModule):
    def __init__(
        self,
        misc_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        predictor_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        physics_config: Mapping[str, Any],
        restoration_config: Mapping[str, Any],
        bridge_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.bridge_config = bridge_config

        self.autoencoder = instantiate_from_config(ae_config)
        self.predictor = instantiate_from_config(predictor_config)
        self.physics_forward = instantiate_from_config(physics_config)
        self.restoration_model = instantiate_from_config(restoration_config)

        self.ae_ckpt_path = ae_config["checkpoint"]
        self.predictor_ckpt_path = bridge_config.get("predictor_checkpoint")
        self.t_learnable = bool(bridge_config.get("t_learnable", False))
        self.t_params = LearnableTParameters(learnable=self.t_learnable)

        self.image_loss_scale = float(bridge_config.get("image_loss_scale", 1.0))
        self.restoration_loss_scale = float(bridge_config.get("restoration_loss_scale", 1.0))
        self.t_anchor_reg_scale = float(bridge_config.get("t_anchor_reg_scale", 0.0))
        self.test_y_channel = bool(bridge_config.get("test_y_channel", True))

        self.consistency_loss = DegradationConsistencyLoss(
            mean_loss_scale=float(bridge_config.get("mean_loss_scale", 1.0)),
            structure_loss_scale=float(bridge_config.get("structure_loss_scale", 0.0)),
            noise_stats_loss_scale=float(bridge_config.get("noise_stats_loss_scale", 0.0)),
            noise_map_loss_scale=float(bridge_config.get("noise_map_loss_scale", 0.0)),
            z_loss_scale=float(bridge_config.get("z_loss_scale", 0.5)),
            z_highfreq_loss_scale=float(bridge_config.get("z_highfreq_loss_scale", 0.0)),
            z_cosine_weight=float(bridge_config.get("z_cosine_weight", 0.2)),
            charbonnier_eps=float(bridge_config.get("charbonnier_eps", 1e-3)),
            lowpass_kernel_size=int(bridge_config.get("lowpass_kernel_size", 21)),
            lowpass_sigma=float(bridge_config.get("lowpass_sigma", 2.0)),
            noise_local_window_size=int(bridge_config.get("noise_local_window_size", 9)),
            noise_map_pool_size=int(bridge_config.get("noise_map_pool_size", 8)),
        )

        self.val_restore_psnr = MeanMetric(dist_sync_on_step=True)
        self.val_deg_psnr = MeanMetric(dist_sync_on_step=True)
        self.register_buffer("t_anchor_positions", torch.zeros(len(LEARNABLE_T_KEYS) + 1, dtype=torch.float32), persistent=False)

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
                "The EndoIR bridge stage requires a rewritten stage-1 checkpoint. "
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

        if "physics_forward" in checkpoint:
            self.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)

        if self.predictor_ckpt_path:
            predictor_checkpoint = torch.load(self.predictor_ckpt_path, map_location="cpu")
            if "predictor" not in predictor_checkpoint:
                raise KeyError(f"Predictor checkpoint missing 'predictor' key: {self.predictor_ckpt_path}")
            self.predictor.load_state_dict(predictor_checkpoint["predictor"], strict=True)
            if "t_params" in predictor_checkpoint:
                self.t_params.load_export_state_dict(predictor_checkpoint["t_params"])

        frozen_module(self.autoencoder)
        frozen_module(self.restoration_model)

    def configure_optimizers(self):
        optim_groups = [{"params": list(self.predictor.parameters())}]

        physics_params = [param for param in self.physics_forward.parameters() if param.requires_grad]
        if physics_params:
            optim_groups.append({"params": physics_params})

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

    def _level_t(self, lq_key: str, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_value = self.t_params.as_dict(device=device, dtype=dtype)[lq_key]
        return t_value.expand(batch_size)

    def t_anchor_loss(self) -> torch.Tensor:
        current = self.t_params.positions_tensor(device=self.device, dtype=torch.float32)
        anchor = self.t_anchor_positions.to(device=current.device, dtype=current.dtype)
        return nn.functional.mse_loss(current[1:], anchor[1:], reduction="mean")

    def _shared_step(self, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        real_hq = batch["GT"]
        real_lq = batch["LQ"]
        lq_keys = batch["lq_key"]

        restored_hq = self.restoration_model(real_lq)
        content_dict = self.autoencoder.encode_content(restored_hq)
        z_content = content_dict["z_c"]

        with torch.no_grad():
            target_z = self.autoencoder.encode_degradation(real_lq)["z_t"]

        t = self.t_params.batch_values(lq_keys, device=real_hq.device, dtype=real_hq.dtype)
        pred_z = self.predictor(z_content=z_content, clean=restored_hq, t=t)["pred_z"]
        physics_out = self.physics_forward(
            clean=restored_hq,
            t=t,
            degradation_latent=pred_z,
            sample_noise=False,
        )

        pred_mean = physics_out.get("degraded_mean", physics_out["degraded"])
        pred_sample = physics_out.get("degraded_learned") if self.consistency_loss.needs_noise_sample else None

        _, consistency_parts = self.consistency_loss(
            pred_mean=pred_mean,
            real_lq=real_lq,
            pred_z=pred_z,
            target_z=target_z,
            pred_sample=pred_sample,
        )

        loss_restore = nn.functional.l1_loss(restored_hq, real_hq)
        total_loss = self.restoration_loss_scale * loss_restore + self.image_loss_scale * consistency_parts["total"]

        if self.t_learnable and self.t_anchor_reg_scale > 0:
            loss_t_anchor = self.t_anchor_loss()
            total_loss = total_loss + self.t_anchor_reg_scale * loss_t_anchor
        else:
            loss_t_anchor = torch.zeros((), device=real_hq.device)

        log_prefix = "train" if mode == "train" else "val"
        pred_deg_for_metric = physics_out.get("degraded_learned", pred_mean)
        self.log_dict(
            {
                f"{log_prefix}/loss": total_loss,
                f"{log_prefix}/restore": loss_restore,
                f"{log_prefix}/mean": consistency_parts["mean"],
                f"{log_prefix}/structure": consistency_parts["structure"],
                f"{log_prefix}/noise_stats": consistency_parts["noise_stats"],
                f"{log_prefix}/noise_map": consistency_parts["noise_map"],
                f"{log_prefix}/z": consistency_parts["z"],
                f"{log_prefix}/z_highfreq": consistency_parts["z_highfreq"],
                f"{log_prefix}/t_anchor": loss_t_anchor,
                f"{log_prefix}/sigma_geo": physics_out["sigma_geo"].mean(),
                f"{log_prefix}/sigma_det": physics_out["sigma_det"].mean(),
                f"{log_prefix}/readout": physics_out["readout_sigma"].mean(),
            },
            prog_bar=(mode == "train"),
            on_step=(mode == "train"),
            on_epoch=True,
            sync_dist=(mode != "train"),
        )

        for key in LEARNABLE_T_KEYS:
            self.log(
                f"{log_prefix}/t_{key.split('_')[1].lower()}",
                self.t_params.as_dict(device=real_hq.device, dtype=real_hq.dtype)[key],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=(mode != "train"),
            )

        if mode == "val":
            restore_psnr = calculate_psnr_pt(
                real_hq.add(1).div(2),
                restored_hq.clamp(-1, 1).add(1).div(2),
                crop_border=0,
                test_y_channel=self.test_y_channel,
            )
            deg_psnr = calculate_psnr_pt(
                real_lq.add(1).div(2),
                pred_deg_for_metric.clamp(-1, 1).add(1).div(2),
                crop_border=0,
                test_y_channel=self.test_y_channel,
            )
            self.val_restore_psnr.update(restore_psnr)
            self.val_deg_psnr.update(deg_psnr)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, mode="val")

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "psnr/restored_hq": self.val_restore_psnr.compute(),
                "psnr/degraded_lq": self.val_deg_psnr.compute(),
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=True,
        )
        self.val_restore_psnr.reset()
        self.val_deg_psnr.reset()
