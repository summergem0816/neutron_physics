from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from piq import LPIPS
from torch import nn
from torchmetrics import MeanMetric

from utils.common import instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt
from utils.neutron_schedule import build_t_tensor_dict


class LitAE(pl.LightningModule):
    def __init__(
        self,
        misc_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        physics_config: Mapping[str, Any] = None,
        scheduler_config: Mapping[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.lpips_loss_scale = ae_config["lpips_loss_scale"]
        self.physics_loss_scale = ae_config.get("physics_loss_scale", 0.0)
        self.global_deg_reg_scale = ae_config.get("global_deg_reg_scale", 0.0)
        self.local_deg_reg_scale = ae_config.get("local_deg_reg_scale", 0.0)
        self.test_y_channel = ae_config["test_y_channel"]

        if self.lpips_loss_scale > 0:
            self.lpips = LPIPS(replace_pooling=True, reduction="none")

        self.autoencoder = instantiate_from_config(model_config)
        if self.misc_config.compile:
            self.autoencoder = torch.compile(self.autoencoder)

        self.physics_forward = instantiate_from_config(physics_config) if physics_config else None

        t_init = build_t_tensor_dict()
        self.t_params = nn.ParameterDict(
            {
                "LQ_50": nn.Parameter(t_init["LQ_50"].clone(), requires_grad=False),
                "LQ_30": nn.Parameter(t_init["LQ_30"].clone(), requires_grad=False),
                "LQ_20": nn.Parameter(t_init["LQ_20"].clone(), requires_grad=False),
                "LQ_10": nn.Parameter(t_init["LQ_10"].clone(), requires_grad=False),
            }
        )

        self.training_step = self.training_random_res
        self.val_recon_psnr = nn.ModuleDict(
            {
                res: MeanMetric(dist_sync_on_step=True)
                for res in ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]
            }
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None and self.trainer.logger is not None and hasattr(self.trainer.logger, "version"):
            self.version = self.trainer.logger.version
        else:
            self.version = "temp"

    def configure_optimizers(self):
        optimizer = instantiate_from_config_with_arg(
            self.optimizer_config,
            [{"params": self.autoencoder.parameters()}],
        )
        optim_config = {"optimizer": optimizer}
        if self.scheduler_config:
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate_from_config_with_arg(self.scheduler_config, optimizer),
                "interval": "step",
                "frequency": 1,
            }
        return optim_config

    def reconstruction_loss(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x, reduction="mean")

    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    def on_train_batch_start(self, batch, batch_idx):
        x = batch["GT"]
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()

    def training_random_res(self, batch, batch_idx):
        hr = batch["GT"]
        lq = batch["LQ"]
        t = batch["t"]

        content_dict = self.autoencoder.encode_content(hr)
        degradation_dict = self.autoencoder.encode_degradation(lq)
        z_t = degradation_dict["z_t"]
        x_rec = self.autoencoder.reconstruct_from_content_degradation(z_t, content_dict["features"])

        loss_recon = self.reconstruction_loss(x_rec, lq)
        loss = loss_recon
        logs = {"recon:": loss_recon}

        if self.lpips_loss_scale > 0:
            loss_lpips = self.lpips(x_rec * 0.5 + 0.5, lq * 0.5 + 0.5).mean()
            logs["lpips"] = loss_lpips
            loss += self.lpips_loss_scale * loss_lpips

        if self.global_deg_reg_scale > 0:
            loss_global_reg = degradation_dict["global_code"].pow(2).mean()
            logs["global_reg"] = loss_global_reg
            loss += self.global_deg_reg_scale * loss_global_reg

        if self.local_deg_reg_scale > 0:
            loss_local_reg = degradation_dict["local_code"].pow(2).mean()
            logs["local_reg"] = loss_local_reg
            loss += self.local_deg_reg_scale * loss_local_reg

        if self.physics_forward is not None and self.physics_loss_scale > 0:
            physics_out = self.physics_forward(hr, t, degradation_latent=z_t, sample_noise=False)
            loss_phys = self.reconstruction_loss(physics_out["degraded"], lq)
            logs["physics"] = loss_phys
            logs["sigma_geo"] = physics_out["sigma_geo"].mean()
            logs["sigma_det"] = physics_out["sigma_det"].mean()
            logs["particle_eff"] = physics_out["particle_count_effective"].mean()
            loss += self.physics_loss_scale * loss_phys

        self.log_dict(logs, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        warmup_iter = self.misc_config.warmup
        if warmup_iter is None or warmup_iter <= 0:
            return

        if self.trainer.global_step < warmup_iter:
            base_lr = self.optimizer_config.params.lr
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warmup_iter)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * base_lr

    def validation_step(self, batch, batch_idx):
        resolutions = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]
        content_dict = self.autoencoder.encode_content(batch["GT"])

        for res in resolutions:
            img = batch[res]
            degradation_dict = self.autoencoder.encode_degradation(img)
            recon = torch.clamp(
                self.autoencoder.reconstruct_from_content_degradation(
                    degradation_dict["z_t"],
                    content_dict["features"],
                ),
                -1,
                1,
            ).add(1).div(2)

            img_norm = img.add(1).div(2)
            psnr_val = calculate_psnr_pt(
                img_norm,
                recon,
                crop_border=0,
                test_y_channel=self.test_y_channel,
            )
            self.val_recon_psnr[res].update(psnr_val)

    def on_validation_epoch_end(self):
        logs = {}
        for res, metric in self.val_recon_psnr.items():
            logs[res] = metric.compute()
            metric.reset()

        logs["t_50"] = self.t_params["LQ_50"].detach()
        logs["t_30"] = self.t_params["LQ_30"].detach()
        logs["t_20"] = self.t_params["LQ_20"].detach()
        logs["t_10"] = self.t_params["LQ_10"].detach()

        self.log_dict(
            logs,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=True,
        )
