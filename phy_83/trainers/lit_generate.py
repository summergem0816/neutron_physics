import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
import numpy as np
from utils.common import instantiate_from_config
from scipy import integrate
from utils.interpolation import natural_cubic_spline_interpolate, linear_interpolate
from utils.neutron_schedule import export_t_map_from_state_dict


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def _crop_to_size(x: torch.Tensor, size):
    height, width = size
    return x[..., :height, :width]


class LitGenerator(pl.LightningModule):
    def __init__(
            self,
            version,
            root_dir,
            camera_type,
            ae_config,
            rf_config=None,
            physics_config=None,
            mode='flow'           #'ae_interp'
        ):
        super().__init__()
        self.save_hyperparameters()
        self.version     = version
        self.root_dir    = root_dir
        self.camera_type = camera_type
        self.mode        = mode
        self.physics_forward = instantiate_from_config(physics_config) if physics_config else None

        # Autoencoder
        autoencoder = instantiate_from_config(ae_config)
        self.content_encoder = autoencoder.content_encoder
        self.global_degradation_encoder = autoencoder.global_degradation_encoder
        self.local_degradation_encoder = autoencoder.local_degradation_encoder
        self.degradation_compressor = autoencoder.degradation_compressor
        self.decoder = autoencoder.decoder
        self.use_skip = autoencoder.use_skip
        self.ae_ckpt_path = ae_config['checkpoint']

        # Flow-model
        if self.mode == 'flow':
            assert rf_config is not None, "needs rf_config."
            self.flow_model = instantiate_from_config(rf_config)
            self.rf_ckpt_path = rf_config['checkpoint']
        
        self.total_time = 0
        self.n = 0
        self.t_map = export_t_map_from_state_dict(None, device="cpu", dtype=torch.float32)

    def setup(self, stage):
        ae_ckpt = torch.load(self.ae_ckpt_path, map_location=self.device)
        required_keys = [
            "content_encoder",
            "global_degradation_encoder",
            "local_degradation_encoder",
            "degradation_compressor",
            "decoder",
        ]
        missing_keys = [key for key in required_keys if key not in ae_ckpt]
        if missing_keys:
            raise ValueError(
                "The generation script requires a rewritten stage-1 checkpoint. "
                f"Missing keys in AE checkpoint: {missing_keys}"
            )
        self.content_encoder.load_state_dict(ae_ckpt["content_encoder"])
        self.global_degradation_encoder.load_state_dict(ae_ckpt["global_degradation_encoder"])
        self.local_degradation_encoder.load_state_dict(ae_ckpt["local_degradation_encoder"])
        self.degradation_compressor.load_state_dict(ae_ckpt["degradation_compressor"])
        self.decoder.load_state_dict(ae_ckpt["decoder"])
        if self.physics_forward is not None and "physics_forward" in ae_ckpt:
            self.physics_forward.load_state_dict(ae_ckpt["physics_forward"], strict=False)
        self.t_map = export_t_map_from_state_dict(ae_ckpt.get("t_params"), device="cpu", dtype=torch.float32)
        self.content_encoder.eval().to(memory_format=torch.channels_last)
        self.global_degradation_encoder.eval().to(memory_format=torch.channels_last)
        self.local_degradation_encoder.eval().to(memory_format=torch.channels_last)
        self.degradation_compressor.eval().to(memory_format=torch.channels_last)
        self.decoder.eval().to(memory_format=torch.channels_last)

        if self.mode == 'flow':
            rf_ckpt = torch.load(self.rf_ckpt_path, map_location=self.device)
            self.flow_model.load_state_dict(rf_ckpt['model'])
            if "t_params" in rf_ckpt:
                self.t_map = export_t_map_from_state_dict(rf_ckpt["t_params"], device="cpu", dtype=torch.float32)
            self.flow_model.eval()

    @torch.no_grad()
    def rk45_sampler(self, z, t_eval, T=1):
        rtol, atol = 1e-5, 1e-5
        method, eps = 'RK45', 1e-3
        def to_flat(x): return x.detach().cpu().numpy().reshape(-1)
        def from_flat(x, shape): return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x_flat):
            x = from_flat(x_flat, z.shape).to(z.device).float()
            vec_t = torch.full((x.size(0),), t, device=x.device)
            drift = self.flow_model(x, vec_t * 999)
            return to_flat(drift)

        sol = integrate.solve_ivp(
            ode_func, (eps, T), to_flat(z), rtol=rtol, atol=atol,
            method=method, t_eval=t_eval, vectorized=True
        )
        outputs = [
            torch.tensor(sol.y[:, i])
                 .reshape(z.shape).to(z.device).float()
            for i in range(len(sol.t))
        ]
        return outputs, sol.nfev

    def on_predict_epoch_start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        raw = batch['prefix']
        if isinstance(raw, (list, tuple)): 
            raw = raw[0]
        prefix = (raw.decode() if isinstance(raw, (bytes, bytearray)) 
                else raw.item() if torch.is_tensor(raw) else raw)

        out_base = os.path.join(self.root_dir, self.camera_type, f"version{self.version}")

        img_gt = batch['GT']           # [-1,1]
        hr     = (img_gt + 1) / 2      # [0,1]

        if self.mode == 'flow':
            img_padded, original_size = _pad_to_multiple(img_gt, multiple=16)
            z, feature = self.content_encoder(img_padded)

            neutron_levels = [
                ("LQ_50", float(self.t_map["LQ_50"].item())),
                ("LQ_30", float(self.t_map["LQ_30"].item())),
                ("LQ_20", float(self.t_map["LQ_20"].item())),
                ("LQ_10", float(self.t_map["LQ_10"].item())),
            ]
            t_eval = np.asarray([item[1] for item in neutron_levels], dtype=np.float32)
            preds, _ = self.rk45_sampler(z, t_eval)

            for (level_name, t_value), p in zip(neutron_levels, preds):
                if self.physics_forward is not None:
                    t_tensor = torch.full((img_gt.shape[0],), float(t_value), device=img_gt.device)
                    lr = self.physics_forward(img_padded, t_tensor, degradation_latent=p, sample_noise=False)["degraded"]
                else:
                    lr = self.decoder(p, feature)
                lr.clamp_(-1, 1)
                lr = _crop_to_size(lr, original_size)

                out_dir = os.path.join(out_base, level_name)
                os.makedirs(out_dir, exist_ok=True)

                save_image(hr[0], os.path.join(out_dir, f"{prefix}_HR.png"))

                save_image(((lr + 1) / 2)[0],
                           os.path.join(out_dir, f"{prefix}_{level_name}.png"))



        elif self.mode == 'ae_interp':
            img_padded, original_size = _pad_to_multiple(batch['GT'], multiple=2)
            z_hr, feature_hr = self.content_encoder(img_padded)
            latent_scales = {
                1.0: z_hr,
                2.0: self.degradation_compressor(
                    self.global_degradation_encoder(batch['LQ_50']),
                    self.local_degradation_encoder(batch['LQ_50']),
                ),
                4.0: self.degradation_compressor(
                    self.global_degradation_encoder(batch['LQ_10']),
                    self.local_degradation_encoder(batch['LQ_10']),
                ),
            }

            # control_points shape: (3, B, C, H, W)
            control_points = torch.stack([
                latent_scales[1.0],
                latent_scales[2.0],
                latent_scales[4.0]
            ], dim=0)
            control_positions = torch.tensor([0.0, 1/3, 1.0],
                                             device=control_points.device)

            B = control_points.shape[1]
            resolutions = np.asarray([1.0, 1.35, 1.65, 1.80, 2.0], dtype=np.float32)
            t_values = np.asarray(
                [
                    0.0,
                    float(self.t_map["LQ_50"].item()),
                    float(self.t_map["LQ_30"].item()),
                    float(self.t_map["LQ_20"].item()),
                    float(self.t_map["LQ_10"].item()),
                ],
                dtype=np.float32,
            )

            for scale, t_val in zip(resolutions[1:], t_values[1:]):
                t_tensor = torch.full((B,), float(t_val),
                                      device=control_points.device)
                z_s, deriv, second_deriv, third_deriv = natural_cubic_spline_interpolate(
                    control_points, t_tensor, control_positions
                )
                # z_s, deriv, second_deriv, third_deriv = linear_interpolate(
                #     control_points, t_tensor, control_positions
                # )

                s_str  = f"{scale:.10f}".rstrip('0').rstrip('.')
                out_dir = os.path.join(out_base, s_str)
                os.makedirs(out_dir, exist_ok=True)

                save_image(hr[0], os.path.join(out_dir, f"{prefix}_HR.png"))

                lr = torch.clamp(self.decoder(z_s, feature_hr), -1, 1)
                lr = _crop_to_size(lr, original_size)

                save_image(((lr + 1) / 2)[0],
                           os.path.join(out_dir, f"{prefix}_LR{s_str}.png"))


        else:
            raise ValueError(f"Unknown mode: {self.mode}")
