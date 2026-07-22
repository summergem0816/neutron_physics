from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.neutron_deg import NeutronTrajectoryDataset
from models.ae import Autoencoder
from models.neutron.physics_forward import NeutronPhysicalForward
from utils.neutron_schedule import export_t_map_from_state_dict


def _unwrap_name(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if torch.is_tensor(value):
        return value.item()
    return value


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8):
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AE .pth checkpoint")
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset root containing HQ and LQ")
    parser.add_argument("--output", type=str, required=True, help="Directory to save visualization results")
    parser.add_argument("--hq_subdir", type=str, default="HQ")
    parser.add_argument("--lq_subdir", type=str, default="LQ")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_noise", action="store_true", help="Enable stochastic Poisson/readout sampling")
    parser.add_argument("--base_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument("--use_skip", action="store_true", default=True)
    parser.add_argument("--no_use_skip", action="store_false", dest="use_skip")
    parser.add_argument("--global_deg_dim", type=int, default=256)
    parser.add_argument("--local_deg_dim", type=int, default=4)
    parser.add_argument("--encoder_kernel_size", type=int, default=13)
    parser.add_argument("--generator_kernel_size", type=int, default=3)
    parser.add_argument("--n_feats_map", type=int, default=16)
    parser.add_argument("--geo_kernel_size", type=int, default=21)
    parser.add_argument("--det_kernel_size", type=int, default=15)
    parser.add_argument("--scatter_kernel_size", type=int, default=81)
    parser.add_argument("--theta_div_deg", type=float, default=1.0)
    parser.add_argument("--object_detector_distance_mm", type=float, default=50.0)
    parser.add_argument("--detector_pitch_mm", type=float, default=0.66)
    parser.add_argument("--detector_pixel_width_mm", type=float, default=0.6)
    parser.add_argument("--scintillator_sigma_px_init", type=float, default=0.35)
    parser.add_argument("--flux_ref_max", type=float, default=3.0e8)
    parser.add_argument("--particle_count_min", type=float, default=1.0e7)
    parser.add_argument("--particle_count_max", type=float, default=3.0e8)
    parser.add_argument("--scatter_strength_max", type=float, default=0.08)
    parser.add_argument("--readout_sigma_max", type=float, default=0.01)
    parser.add_argument("--latent_modulation_scale", type=float, default=0.15)
    parser.add_argument("--trainable_blur", action="store_true", default=True)
    parser.add_argument("--no_trainable_blur", action="store_false", dest="trainable_blur")
    parser.add_argument("--trainable_noise_heads", action="store_true", default=True)
    parser.add_argument("--no_trainable_noise_heads", action="store_false", dest="trainable_noise_heads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = NeutronTrajectoryDataset(
        dataroot=args.dataroot,
        hq_subdir=args.hq_subdir,
        lq_subdir=args.lq_subdir,
        augmentation=False,
        test=True,
        patch_size=None,
        preload=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    autoencoder = Autoencoder(
        in_dim=args.in_dim,
        base_dim=args.base_dim,
        latent_dim=args.latent_dim,
        use_skip=args.use_skip,
        global_deg_dim=args.global_deg_dim,
        local_deg_dim=args.local_deg_dim,
        encoder_kernel_size=args.encoder_kernel_size,
        generator_kernel_size=args.generator_kernel_size,
        n_feats_map=args.n_feats_map,
    ).to(device)
    physics_forward = NeutronPhysicalForward(
        geo_kernel_size=args.geo_kernel_size,
        det_kernel_size=args.det_kernel_size,
        scatter_kernel_size=args.scatter_kernel_size,
        theta_div_deg=args.theta_div_deg,
        object_detector_distance_mm=args.object_detector_distance_mm,
        detector_pitch_mm=args.detector_pitch_mm,
        detector_pixel_width_mm=args.detector_pixel_width_mm,
        scintillator_sigma_px_init=args.scintillator_sigma_px_init,
        flux_ref_max=args.flux_ref_max,
        particle_count_min=args.particle_count_min,
        particle_count_max=args.particle_count_max,
        scatter_strength_max=args.scatter_strength_max,
        readout_sigma_max=args.readout_sigma_max,
        latent_modulation_scale=args.latent_modulation_scale,
        latent_channels=args.latent_dim,
        trainable_blur=args.trainable_blur,
        trainable_noise_heads=args.trainable_noise_heads,
    ).to(device)

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

    required_keys = [
        "content_encoder",
        "global_degradation_encoder",
        "local_degradation_encoder",
        "degradation_compressor",
        "physics_forward",
    ]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            "The physics inference script requires a rewritten stage-1 checkpoint with physics weights. "
            f"Missing keys: {missing_keys}"
        )

    autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
    autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
    autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
    autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
    physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)

    autoencoder.eval()
    physics_forward.eval()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    resolution_order = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]
    learned_t_map = export_t_map_from_state_dict(checkpoint.get("t_params"), device=device, dtype=torch.float32)
    t_values = torch.stack(
        [
            learned_t_map["GT"],
            learned_t_map["LQ_50"],
            learned_t_map["LQ_30"],
            learned_t_map["LQ_20"],
            learned_t_map["LQ_10"],
        ],
        dim=0,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_samples is not None and batch_idx >= args.max_samples:
                break

            gt = batch["GT"].to(device)
            gt_padded, _ = _pad_to_multiple(gt, multiple=2)

            prefix = str(_unwrap_name(batch["prefix"]))
            sample_dir = output_root / prefix
            sample_dir.mkdir(parents=True, exist_ok=True)

            save_image(gt.add(1).div(2), sample_dir / "00_GT.png")

            for order_idx, key in enumerate(resolution_order):
                img = batch[key].to(device)
                img_padded, img_size = _pad_to_multiple(img, multiple=2)
                degradation_dict = autoencoder.encode_degradation(img_padded)
                t = t_values[order_idx].to(device=gt.device, dtype=gt.dtype).expand(gt.shape[0])
                physics_out = physics_forward(
                    gt_padded,
                    t,
                    degradation_latent=degradation_dict["z_t"],
                    sample_noise=args.sample_noise,
                )
                phys = torch.clamp(physics_out["degraded"], -1, 1)
                phys = _crop_to_size(phys, img_size)

                if key == "GT":
                    save_image(phys.add(1).div(2), sample_dir / "01_PHYS_GT.png")
                    continue

                offset = 2 + (order_idx - 1) * 2
                save_image(img.add(1).div(2), sample_dir / f"{offset:02d}_{key}.png")
                save_image(phys.add(1).div(2), sample_dir / f"{offset + 1:02d}_PHYS_{key}.png")

    print(f"Saved AE physics-forward results to {output_root}")


if __name__ == "__main__":
    main()
