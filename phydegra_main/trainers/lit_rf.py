import copy
from collections import OrderedDict
from utils.common import frozen_module

import torch
from torch import nn
import torchmetrics
import wandb
from utils.common import instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt, calculate_ssim_pt, calculate_kld, calculate_alkd
from utils.interpolation import linear_interpolate, natural_cubic_spline_interpolate
from utils.t_dist import ExponentialPDF, sample_t, sample_t_uniform
from utils.noise import DiagonalGaussianDistribution
from utils.neutron_schedule import LearnableTParameters
from torchvision.utils import make_grid
from torchvision.utils import save_image
import pytorch_lightning as pl
from scipy import integrate
# from losses.distribution_loss import Gaussian
# from torchvision.transforms import ToPILImage
from trainers.lit_ema import LitEma
from contextlib import contextmanager
from typing import Mapping, Any, List, Optional, Tuple, Union
import torch.nn.functional as F

# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
import math

import numpy as np
import random

import torch.distributed as dist

import os

from piq import LPIPS


def logit_normal_sampling(t, m=0, s=1):
    return (1 / (s*np.sqrt(2*np.pi))) * (1 / (t*(1-t))) * torch.exp(-((torch.log2(t/(1-t)-m))**2/2*s**2))


class LitRectifiedFlow(pl.LightningModule):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        rf_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        # loss_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any],
        compile: bool,
        sampler_type: str,
        sample_N: int,
        physics_config: Mapping[str, Any] = None,
        use_ema: bool = False,
        ):
        super().__init__()

        self.save_hyperparameters()
        
        autoencoder = instantiate_from_config(ae_config)
        self.physics_forward = instantiate_from_config(physics_config) if physics_config else None
        
        self.content_encoder = autoencoder.content_encoder
        self.global_degradation_encoder = autoencoder.global_degradation_encoder
        self.local_degradation_encoder = autoencoder.local_degradation_encoder
        self.degradation_compressor = autoencoder.degradation_compressor
        self.decoder = autoencoder.reconstruction_decoder
        self.use_skip = autoencoder.use_skip

        self.t_learnable = rf_config['t_learnable']
        self.t_params = LearnableTParameters(learnable=self.t_learnable)
        self.t_anchor_reg_scale = rf_config.get('t_anchor_reg_scale', 0.0)
        self.t_lr_scale = rf_config.get('t_lr_scale', 1.0)

        self.ae_ckpt_path = ae_config['checkpoint']

        # instantiate model
        self.model = instantiate_from_config(model_config)
        if compile:
            self.model = torch.compile(self.model)
        # self.loss = instantiate_from_config(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        # only for euler
        self.sample_N = sample_N
        print('Number of sampling steps:', self.sample_N)

        if sampler_type == 'rk45':
            self.sampler = self.rk45_sampler
        elif sampler_type == 'euler':
            self.sampler = self.euler_sampler
        elif sampler_type == 'dopri5':
            self.sampler = self.dopri5_sampler
        else:
            raise NotImplementedError()
        
        self.T = 1

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.train_resolutions = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]
        self.register_buffer(
            "t_anchor_positions",
            torch.zeros(len(self.train_resolutions), dtype=torch.float32),
            persistent=False,
        )
        self.interp_method = rf_config['interp_method']
        if self.interp_method == 'linear':
            self.interp_fn = linear_interpolate
        elif self.interp_method == 'cubic_spline':
            self.interp_fn = natural_cubic_spline_interpolate
        else:
            raise NotImplementedError()
        
        self.loss_type = rf_config['loss_type']
        if self.loss_type == 'l2':
            self.loss = self.l2_loss
        else:
            raise NotImplementedError()

        self.lpips = LPIPS(replace_pooling=True, reduction="none")
        self.flow_loss_scale = rf_config.get('flow_loss_scale', 1.0)
        self.lpips_scale = rf_config['lpips_scale']
        self.lpips_weighting = rf_config['lpips_weighting']
        self.z_loss_scale = rf_config.get('z_loss_scale', 1.0)
        self.z_cosine_weight = rf_config.get('z_cosine_weight', 0.2)
        self.physics_loss_scale = rf_config.get('physics_loss_scale', 1.0)
        
        self.test_y_channel = rf_config['test_y_channel']

        self.resolutions = ["LQ_50", "LQ_30", "LQ_20", "LQ_10"]
        self.psnr_metrics = torch.nn.ModuleDict({
            res: torchmetrics.MeanMetric() for res in self.resolutions
        })
        self.nfe_metric = torchmetrics.MeanMetric()

        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')


        
    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None and self.trainer.logger is not None and hasattr(self.trainer.logger, "version"):
            self.version = self.trainer.logger.version
        else:
            self.version = "temp"
        
        autoencoder_ckpt = torch.load(self.ae_ckpt_path, map_location=self.device)
        required_keys = [
            "content_encoder",
            "global_degradation_encoder",
            "local_degradation_encoder",
            "degradation_compressor",
            "decoder",
            "t_params",
        ]
        missing_keys = [key for key in required_keys if key not in autoencoder_ckpt]
        if missing_keys:
            raise ValueError(
                "The stage-2 model requires a stage-1 checkpoint from the rewritten neutron pipeline. "
                f"Missing keys in AE checkpoint: {missing_keys}"
            )
        self.content_encoder.load_state_dict(autoencoder_ckpt["content_encoder"])
        self.global_degradation_encoder.load_state_dict(autoencoder_ckpt["global_degradation_encoder"])
        self.local_degradation_encoder.load_state_dict(autoencoder_ckpt["local_degradation_encoder"])
        self.degradation_compressor.load_state_dict(autoencoder_ckpt["degradation_compressor"])
        self.decoder.load_state_dict(autoencoder_ckpt["decoder"])
        self.t_params.load_export_state_dict(autoencoder_ckpt["t_params"])
        with torch.no_grad():
            anchor_positions = self.t_params.positions_tensor(device=self.device, dtype=torch.float32)
            self.t_anchor_positions.copy_(anchor_positions.detach().to(self.t_anchor_positions))
        if self.physics_forward is not None and "physics_forward" in autoencoder_ckpt:
            self.physics_forward.load_state_dict(autoencoder_ckpt["physics_forward"], strict=False)

        for module in (
            self.content_encoder,
            self.global_degradation_encoder,
            self.local_degradation_encoder,
            self.degradation_compressor,
            self.decoder,
        ):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


    def configure_optimizers(self):
        optim_params = [{'params': self.model.parameters()},]

        if self.t_learnable:
            t_group = {'params': self.t_params.parameters()}
            base_lr = self.optimizer_config.get('params', {}).get('lr')
            if base_lr is not None:
                t_group['lr'] = float(base_lr) * float(self.t_lr_scale)
            optim_params.append(t_group)

        optimizer = instantiate_from_config_with_arg(self.optimizer_config, optim_params)

        learning_rate_scheduler = instantiate_from_config_with_arg(
            self.scheduler_config, optimizer)
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": learning_rate_scheduler,
                    "interval": 'step',
                    "frequency": 1,},}

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x, sampler_type='euler', sample_N=None, reverse=False, set_grad=False):
        if sampler_type == 'rk45':
            x, nfev = self.rk45_sampler(x, reverse_t=reverse)
        elif sampler_type == 'euler':
            x, nfev = self.euler_sampler(x, reverse_t=reverse, sample_N=sample_N, set_grad=set_grad)
        else:
            raise NotImplementedError()

        return x, nfev
    
    
    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1
        
    @property
    def t_positions(self):
        dtype = next(self.model.parameters()).dtype
        return self.t_params.positions_tensor(device=self.device, dtype=dtype)


    def on_train_batch_start(self, batch, batch_idx):
        x = batch['GT']
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()
    
    def training_step(self, batch, batch_idx):
        z_list = []
        x_list = []
        feature_hr = None
        gt_img = batch["GT"]

        for res in self.train_resolutions:
            img = batch[res]

            ### hr_skip ###
            if res == "GT":
                z_content, feature = self.content_encoder(img)
                z = z_content
                feature_hr = feature
            else:
                global_code = self.global_degradation_encoder(img)
                local_code = self.local_degradation_encoder(img)
                z = self.degradation_compressor(global_code, local_code)
            # z = self.encoder(img)
            ### hr_skip ###

            z_list.append(z)
            x_list.append(img)
        
        z_tensor = torch.stack(z_list, dim=0)
        x_tensor = torch.stack(x_list, dim=0)

        loss = self.loss_fn(
            latents=z_tensor, 
            imgs=x_tensor, 
            feature_hr = feature_hr, 
            clean_img = gt_img,
        )

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    
    def l2_loss(self, x, y):
        return torch.mean((x - y) ** 2, dim=tuple(range(1, x.dim())))

    def t_anchor_loss(self, t_position: torch.Tensor) -> torch.Tensor:
        anchor = self.t_anchor_positions.to(device=t_position.device, dtype=t_position.dtype)
        return F.mse_loss(t_position[1:], anchor[1:], reduction='mean')

    # Direct latent regression version kept for later ablation if needed.
    # def latent_alignment_loss_direct(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    #     return self.loss(pred_z, target_z)

    def latent_alignment_loss_weighted(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        diff_loss = self.loss(pred_z, target_z)
        pred_flat = pred_z.flatten(1)
        target_flat = target_z.flatten(1)
        cosine_penalty = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1)
        return diff_loss + self.z_cosine_weight * cosine_penalty

    def loss_fn(self, latents, imgs, feature_hr, clean_img, eps=1e-3):
        t_position = self.t_positions
        t = sample_t_uniform(latents[0].shape[0]).to(latents[0].device)

        interp, deriv, second_deriv, third_deriv = self.interp_fn(latents, t, t_position)

        pred = self.model(interp, t * 999)

        loss_flow = self.loss(deriv, pred)
        total_loss = self.flow_loss_scale * loss_flow
        logs = {'flow:': loss_flow.mean()}

        if self.t_learnable and self.t_anchor_reg_scale > 0:
            loss_t_anchor = self.t_anchor_loss(t_position)
            logs['t_anchor'] = loss_t_anchor
            total_loss += self.t_anchor_reg_scale * loss_t_anchor

        if self.lpips_scale > 0:
            idx = torch.searchsorted(t_position, t, right=True)
            idx = torch.clamp(idx, max=t_position.shape[0] - 1)
            t_right = t_position[idx]
            t_left = t_position[idx-1]
            delta_t = (t_right - t).view(-1, 1, 1, 1)

            if self.interp_method == 'linear':
                pred_z = interp + delta_t * pred
            elif self.interp_method == 'cubic_spline':
                # 3rd-order Taylor approximation
                pred_z = (
                    interp
                    + delta_t * pred
                    + 0.5 * (delta_t ** 2) * second_deriv
                    + (1.0 / 6.0) * (delta_t ** 3) * third_deriv
                )
            else:
                raise ValueError(f"Unknown interp_method: {self.interp_method}")

            batch_indices = torch.arange(imgs.shape[1], device=imgs.device)
            target_z = latents[idx, batch_indices, :, :, :]
            x = imgs[idx, batch_indices, :, :, :]

            if self.z_loss_scale > 0:
                loss_z = self.latent_alignment_loss_weighted(pred_z, target_z)
                logs['z_latent'] = loss_z.mean()
                total_loss += self.z_loss_scale * loss_z

            if self.physics_forward is not None:
                pred_x = self.physics_forward(clean_img, t, degradation_latent=pred_z, sample_noise=False)["degraded"]
            else:
                pred_x = self.decoder(pred_z, feature_hr)

            delta_t1 = (t_right - t).clamp(min=eps)
            seg_len = (t_right - t_left).clamp(min=eps)

            r = delta_t1 / seg_len

            lpips_weight = 1.0 / (r + eps) if self.lpips_weighting else 1.0
            
            loss_lpips = self.lpips(pred_x * 0.5 + 0.5, x * 0.5 + 0.5) * lpips_weight
            logs['lpips'] = loss_lpips.mean()
            total_loss += self.lpips_scale * loss_lpips

            if self.physics_forward is not None and self.physics_loss_scale > 0:
                loss_phys = self.loss(pred_x, x)
                logs['physics'] = loss_phys.mean()
                total_loss += self.physics_loss_scale * loss_phys

        loss = torch.mean(total_loss)

        self.log_dict(logs, prog_bar=True)

        return loss
    
    

    def on_validation_epoch_start(self):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)

    def validation_step(self, batch, batch_idx):
        img = batch["GT"]

        ### hr_skip ###
        z, feature_hr = self.content_encoder(img)
        # z = self.encoder(img)
        ### hr_skip ###


        t_eval = self.t_positions[1:].detach().cpu().numpy()

        pred, nfev_forward = self.sampler(z, t_eval=t_eval)

        global_indices = batch["img_idx"] + 1

        for resolution in self.resolutions:
            LR = batch[resolution]
            LR = (LR + 1) / 2

            z_LR_pred = pred[resolution]

            if self.physics_forward is not None:
                t_index = self.train_resolutions.index(resolution)
                t_value = torch.full((img.shape[0],), float(self.t_positions[t_index].item()), device=img.device)
                LR_pred = self.physics_forward(img, t_value, degradation_latent=z_LR_pred, sample_noise=False)["degraded"]
            else:
                LR_pred = self.decoder(z_LR_pred, feature_hr)

            LR_pred_clipped = torch.clamp(LR_pred, -1, 1)
            LR_pred_clipped = (LR_pred_clipped + 1) / 2

            psnr = calculate_psnr_pt(LR, LR_pred_clipped, crop_border=0, test_y_channel=self.test_y_channel)
            self.psnr_metrics[resolution].update(psnr)


            save_dir = os.path.join("results", "rf", resolution)
            os.makedirs(save_dir, exist_ok=True)
            for i in range(LR_pred_clipped.shape[0]):
                sample_idx = global_indices[i] if isinstance(global_indices, list) else int(global_indices[i].item())
                save_path = os.path.join(save_dir, f"val_{sample_idx}_{resolution}.png")
                save_image(LR_pred_clipped[i], save_path)
        

        self.nfe_metric.update(nfev_forward)

    def on_validation_epoch_end(self):
        log_dict = {}
        for res in self.resolutions:
            log_dict[f'{res}_psnr'] = self.psnr_metrics[res].compute()
            self.psnr_metrics[res].reset()

        log_dict['nfe'] = self.nfe_metric.compute()
        self.nfe_metric.reset()
        log_dict['t_50'] = self.t_positions[1].detach()
        log_dict['t_30'] = self.t_positions[2].detach()
        log_dict['t_20'] = self.t_positions[3].detach()
        log_dict['t_10'] = self.t_positions[4].detach()
        if self.t_anchor_reg_scale > 0:
            log_dict['t_anchor_drift'] = self.t_anchor_loss(self.t_positions).detach()

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=True
        )

        if self.use_ema:
            self.model_ema.restore(self.model.parameters())


    @torch.no_grad()
    def rk45_sampler(self, z, reverse_t=False, t_eval=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code z.
        Returns:
        samples, number of function evaluations.
        """

        rtol=1e-5
        atol=1e-5
        method='RK45'
        eps=1e-3

        x = z

        def to_flattened_numpy(x):
            """Flatten a torch tensor x and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given shape from a flattened numpy array x."""
            return torch.from_numpy(x.reshape(shape))


        def ode_func(t, x):
            x = from_flattened_numpy(x, z.shape).to(z.device).type(torch.float32)
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift = self.model(x, vec_t * 999)

            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse_t:
            t_span = (self.T-eps, 0.)
        else:
            t_span = (eps, self.T)
        if t_eval is None:
            t_eval = self.t_positions[1:].detach().cpu().numpy()

        solution = integrate.solve_ivp(ode_func, t_span, to_flattened_numpy(x),
                                        rtol=rtol, atol=atol, method=method, t_eval=t_eval, vectorized=True)
        nfe = solution.nfev

        pred = {}
        pred["GT"] = z
        for idx, resolution in enumerate(self.resolutions):
            pred[resolution] = torch.tensor(solution.y[:, idx]).reshape(z.shape).to(z.device).type(torch.float32)

        return pred, nfe
    
