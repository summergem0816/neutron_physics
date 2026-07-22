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
from utils.neutron_schedule import export_t_map_from_state_dict


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return img


def _pad_to_multiple(x: torch.Tensor, multiple: int = 2):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def _resize_latent(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min).clamp_min(1e-8)


def _latent_mean_map(z: torch.Tensor) -> torch.Tensor:
    return z.mean(dim=1, keepdim=True)


def _latent_abs_delta_map(z_ref: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
    return (z_pred - z_ref).abs().mean(dim=1, keepdim=True)


def _save_channel_grid(z: torch.Tensor, path: Path):
    grid = _normalize_map(z[0].unsqueeze(1))
    save_image(grid, path, nrow=grid.shape[0])


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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config)
    ae_config = config.model.params.ae_config
    rf_config = config.model.params.rf_config
    rf_latent_size = int(rf_config.params.config.data.image_size)

    autoencoder = instantiate_from_config(ae_config).to(device)
    flow_model = instantiate_from_config(rf_config).to(device)

    try:
        ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu")
    autoencoder.content_encoder.load_state_dict(ae_ckpt["content_encoder"])

    try:
        rf_ckpt = torch.load(args.rf_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        rf_ckpt = torch.load(args.rf_checkpoint, map_location="cpu")
    flow_model.load_state_dict(rf_ckpt["model"])

    autoencoder.eval()
    flow_model.eval()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    t_state = rf_ckpt.get("t_params", ae_ckpt.get("t_params"))
    t_map = export_t_map_from_state_dict(t_state, device="cpu", dtype=torch.float32)
    neutron_levels = [
        ("LQ_50", float(t_map["LQ_50"].item())),
        ("LQ_30", float(t_map["LQ_30"].item())),
        ("LQ_20", float(t_map["LQ_20"].item())),
        ("LQ_10", float(t_map["LQ_10"].item())),
    ]
    t_eval = np.asarray([item[1] for item in neutron_levels], dtype=np.float32)

    image_paths = sorted([p for p in Path(args.hq_dir).iterdir() if p.is_file()])
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    with torch.no_grad():
        for img_path in image_paths:
            img = _load_rgb_tensor(img_path).to(device)
            img_padded, _ = _pad_to_multiple(img, multiple=2)

            z_gt, _ = autoencoder.content_encoder(img_padded)
            latent_original_size = z_gt.shape[-2:]
            z_for_flow = _resize_latent(z_gt, size=(rf_latent_size, rf_latent_size))
            preds, nfev = rk45_sampler(flow_model, z_for_flow, t_eval=t_eval, device=device)

            sample_dir = output_root / img_path.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(img.add(1).div(2), sample_dir / "00_HQ.png")

            z_gt_mean = _normalize_map(_latent_mean_map(z_gt))
            save_image(z_gt_mean, sample_dir / "01_GT_latent_mean.png")
            _save_channel_grid(z_gt, sample_dir / "02_GT_latent_channels.png")

            stats_lines = [f"sample={img_path.stem}", f"nfev={nfev}"]
            gt_flat = z_gt.flatten(1)
            prev_flat = gt_flat

            for idx, ((level_name, t_value), pred_z) in enumerate(zip(neutron_levels, preds), start=1):
                pred_z = _resize_latent(pred_z, size=latent_original_size)
                pred_flat = pred_z.flatten(1)

                mean_map = _normalize_map(_latent_mean_map(pred_z))
                delta_map = _normalize_map(_latent_abs_delta_map(z_gt, pred_z))

                save_image(mean_map, sample_dir / f"{10 + idx * 3:02d}_{level_name}_latent_mean.png")
                save_image(delta_map, sample_dir / f"{11 + idx * 3:02d}_{level_name}_delta_to_GT.png")
                _save_channel_grid(pred_z, sample_dir / f"{12 + idx * 3:02d}_{level_name}_latent_channels.png")

                l2_to_gt = torch.norm(pred_flat - gt_flat, dim=1).mean().item()
                l1_to_gt = (pred_flat - gt_flat).abs().mean().item()
                cos_to_gt = F.cosine_similarity(pred_flat, gt_flat, dim=1).mean().item()

                l2_to_prev = torch.norm(pred_flat - prev_flat, dim=1).mean().item()
                l1_to_prev = (pred_flat - prev_flat).abs().mean().item()
                cos_to_prev = F.cosine_similarity(pred_flat, prev_flat, dim=1).mean().item()

                stats_lines.append(
                    (
                        f"{level_name} t={t_value:.6f} "
                        f"l2_to_GT={l2_to_gt:.6f} l1_to_GT={l1_to_gt:.6f} cos_to_GT={cos_to_gt:.6f} "
                        f"l2_to_prev={l2_to_prev:.6f} l1_to_prev={l1_to_prev:.6f} cos_to_prev={cos_to_prev:.6f}"
                    )
                )
                prev_flat = pred_flat

            (sample_dir / "latent_stats.txt").write_text("\n".join(stats_lines), encoding="utf-8")

    print(f"Saved RF latent inspection results to {output_root}")


if __name__ == "__main__":
    main()
