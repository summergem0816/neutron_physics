from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.neutron_deg import NeutronTrajectoryDataset
from models.ae import Autoencoder


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

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "content_encoder" in checkpoint:
        autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
        autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
        autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
        autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
    else:
        autoencoder.content_encoder.load_state_dict(checkpoint["encoder"])
    autoencoder.decoder.load_state_dict(checkpoint["decoder"])
    autoencoder.eval()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    resolution_order = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_samples is not None and batch_idx >= args.max_samples:
                break

            gt = batch["GT"].to(device)
            gt_padded, gt_size = _pad_to_multiple(gt, multiple=2)
            content_dict = autoencoder.encode_content(gt_padded)
            feature_hr = content_dict["features"]

            prefix = str(_unwrap_name(batch["prefix"]))
            sample_dir = output_root / prefix
            sample_dir.mkdir(parents=True, exist_ok=True)

            save_image(gt.add(1).div(2), sample_dir / "00_GT.png")

            for order_idx, key in enumerate(resolution_order):
                img = batch[key].to(device)
                img_padded, img_size = _pad_to_multiple(img, multiple=2)
                degradation_dict = autoencoder.encode_degradation(img_padded)
                recon = torch.clamp(
                    autoencoder.reconstruct_from_content_degradation(
                        degradation_dict["z_t"],
                        feature_hr,
                    ),
                    -1,
                    1,
                )
                recon = _crop_to_size(recon, img_size)

                if key == "GT":
                    save_image(recon.add(1).div(2), sample_dir / "01_RECON_GT.png")
                    continue

                offset = 2 + (order_idx - 1) * 2
                save_image(img.add(1).div(2), sample_dir / f"{offset:02d}_{key}.png")
                save_image(recon.add(1).div(2), sample_dir / f"{offset + 1:02d}_RECON_{key}.png")

    print(f"Saved AE reconstructions to {output_root}")


if __name__ == "__main__":
    main()
