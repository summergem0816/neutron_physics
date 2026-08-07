from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import integrate
from torchvision.utils import save_image

from omegaconf import OmegaConf

from utils.common import instantiate_from_config
from utils.neutron_schedule import build_t_map


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


def _pad_to_multiple(x: torch.Tensor, multiple: int = 64):
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


def rk45_sampler(flow_model, z, t_eval, device):
    rtol, atol = 1e-5, 1e-5
    method, eps, T = "RK45", 1e-3, 1.0

    def to_flat(x):
        return x.detach().cpu().numpy().reshape(-1)

    def from_flat(x, shape):
        return torch.from_numpy(x.reshape(shape))

    def ode_func(t, x_flat):
        x = from_flat(x_flat, z.shape).to(device).float()
        vec_t = torch.full((x.size(0),), t, device=device)
        drift = flow_model(x, vec_t * 999)
        return to_flat(drift)

    sol = integrate.solve_ivp(
        ode_func,
        (eps, T),
        to_flat(z),
        rtol=rtol,
        atol=atol,
        method=method,
        t_eval=t_eval,
        vectorized=True,
    )
    outputs = [
        torch.tensor(sol.y[:, i]).reshape(z.shape).to(device).float()
        for i in range(len(sol.t))
    ]
    return outputs, sol.nfev


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generate_neutron.yaml")
    parser.add_argument("--ae_checkpoint", type=str, required=True)
    parser.add_argument("--rf_checkpoint", type=str, required=True)
    parser.add_argument("--hq_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_noise", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config)
    ae_config = config.model.params.ae_config
    rf_config = config.model.params.rf_config
    physics_config = config.model.params.physics_config

    autoencoder = instantiate_from_config(ae_config).to(device)
    physics_forward = instantiate_from_config(physics_config).to(device) if physics_config else None
    flow_model = instantiate_from_config(rf_config).to(device)

    try:
        ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu")

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
            "The RF inference script requires a rewritten stage-1 checkpoint. "
            f"Missing keys in AE checkpoint: {missing_keys}"
        )

    autoencoder.content_encoder.load_state_dict(ae_ckpt["content_encoder"])
    autoencoder.global_degradation_encoder.load_state_dict(ae_ckpt["global_degradation_encoder"])
    autoencoder.local_degradation_encoder.load_state_dict(ae_ckpt["local_degradation_encoder"])
    autoencoder.degradation_compressor.load_state_dict(ae_ckpt["degradation_compressor"])
    autoencoder.decoder.load_state_dict(ae_ckpt["decoder"])
    if physics_forward is not None and "physics_forward" in ae_ckpt:
        physics_forward.load_state_dict(ae_ckpt["physics_forward"], strict=False)

    try:
        rf_ckpt = torch.load(args.rf_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        rf_ckpt = torch.load(args.rf_checkpoint, map_location="cpu")
    flow_model.load_state_dict(rf_ckpt["model"])

    autoencoder.eval()
    flow_model.eval()
    if physics_forward is not None:
        physics_forward.eval()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    t_map = build_t_map()

    neutron_levels = [
        ("LQ_50", t_map["LQ_50"]),
        ("LQ_30", t_map["LQ_30"]),
        ("LQ_20", t_map["LQ_20"]),
        ("LQ_10", t_map["LQ_10"]),
    ]
    t_eval = np.asarray([item[1] for item in neutron_levels], dtype=np.float32)

    hq_dir = Path(args.hq_dir)
    image_paths = sorted([p for p in hq_dir.iterdir() if p.is_file()])
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    with torch.no_grad():
        for img_path in image_paths:
            img = _load_rgb_tensor(img_path).to(device)
            # The new CDD-style content branch outputs a half-resolution latent.
            # Keep even spatial size so local degradation maps align cleanly.
            img_padded, original_size = _pad_to_multiple(img, multiple=2)

            z, feature = autoencoder.content_encoder(img_padded)

            preds, _ = rk45_sampler(flow_model, z, t_eval=t_eval, device=device)

            prefix = img_path.stem
            sample_dir = output_root / prefix
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(img.add(1).div(2), sample_dir / f"{prefix}_HQ.png")

            for (level_name, t_value), pred_z in zip(neutron_levels, preds):
                if physics_forward is not None:
                    t_tensor = torch.full((img.shape[0],), float(t_value), device=device)
                    lr = physics_forward(
                        img_padded,
                        t_tensor,
                        degradation_latent=pred_z,
                        sample_noise=args.sample_noise,
                    )["degraded"]
                else:
                    lr = autoencoder.decoder(pred_z, feature)

                lr = torch.clamp(lr, -1, 1)
                lr = _crop_to_size(lr, original_size)
                save_image(lr.add(1).div(2), sample_dir / f"{prefix}_{level_name}.png")

    print(f"Saved RF inference results to {output_root}")


if __name__ == "__main__":
    main()
