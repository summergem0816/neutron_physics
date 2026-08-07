from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image

from utils.common import instantiate_from_config


LEVEL_KEYS = ["LQ_50", "LQ_30", "LQ_20", "LQ_10"]
PARTICLE_SUFFIXES = {"1k", "2k", "3k", "5k"}


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    gray = np.array(Image.open(path).convert("L"))
    img = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def _split_prefix_and_suffix(stem: str) -> tuple[str, str] | None:
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    if suffix not in PARTICLE_SUFFIXES:
        return None
    return prefix, suffix


def _collect_anchor_image_paths(hq_dir: Path) -> list[Path]:
    all_files = sorted([path for path in hq_dir.iterdir() if path.is_file()])
    anchor_files = []
    generic_files = []

    for path in all_files:
        parsed = _split_prefix_and_suffix(path.stem)
        if parsed is None:
            generic_files.append(path)
            continue
        _, suffix = parsed
        if suffix == "1k":
            anchor_files.append(path)

    if anchor_files:
        return sorted(generic_files + anchor_files)
    return all_files


def _pad_to_multiple(x: torch.Tensor, multiple: int = 2):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def _crop_to_size(x: torch.Tensor, size: tuple[int, int]):
    height, width = size
    return x[..., :height, :width]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/train_neutron_lit_deg_zonly.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hq_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_noise", action="store_true")
    parser.add_argument("--output_mode", type=str, default="learned", choices=["learned", "mean", "sample"])
    parser.add_argument("--readout_sigma_max", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_cfg = OmegaConf.load(args.train_config)
    model_cfg = OmegaConf.load(train_cfg.model.config)
    model = instantiate_from_config(model_cfg).to(device)
    model.setup(stage="predict")
    model.eval()

    try:
        zonly_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        zonly_ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.predictor.load_state_dict(zonly_ckpt["predictor"])
    if "physics_forward" in zonly_ckpt and model.physics_forward is not None:
        model.physics_forward.load_state_dict(zonly_ckpt["physics_forward"], strict=False)

    if args.readout_sigma_max is not None and model.physics_forward is not None:
        model.physics_forward.readout_sigma_max = float(args.readout_sigma_max)

    t_state = zonly_ckpt.get("t_params")
    if t_state is not None:
        model.t_params.load_export_state_dict(t_state)
    t_map = model.t_params.as_dict(device=device, dtype=torch.float32)

    hq_dir = Path(args.hq_dir)
    image_paths = _collect_anchor_image_paths(hq_dir)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if model.physics_forward is not None:
        print(
            f"Using physics noise settings: "
            f"readout_sigma_max={model.physics_forward.readout_sigma_max}, "
            f"scatter_strength_max={model.physics_forward.scatter_strength_max}, "
            f"sample_noise={args.sample_noise}, "
            f"output_mode={args.output_mode}"
        )
    print(f"Using {len(image_paths)} HQ anchors from {hq_dir}")

    with torch.no_grad():
        for image_path in image_paths:
            clean = _load_rgb_tensor(image_path).to(device)
            clean_padded, original_size = _pad_to_multiple(clean, multiple=2)
            content_dict = model.autoencoder.encode_content(clean_padded)
            z_content = content_dict["z_c"]

            sample_dir = output_root / image_path.stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(clean.add(1).div(2), sample_dir / f"{image_path.stem}_GT.png")

            metadata = {}
            for key in LEVEL_KEYS:
                t = t_map[key].expand(clean.shape[0]).to(device=device, dtype=clean.dtype)
                pred_z = model.predictor(z_content=z_content, clean=clean_padded, t=t)["pred_z"]
                run_sample_noise = bool(args.sample_noise or args.output_mode == "sample")
                physics_out = model.physics_forward(
                    clean=clean_padded,
                    t=t,
                    degradation_latent=pred_z,
                    sample_noise=run_sample_noise,
                )
                if args.output_mode == "learned":
                    output_tensor = physics_out.get("degraded_learned", physics_out["degraded"])
                elif args.output_mode == "mean":
                    output_tensor = physics_out.get("degraded_mean", physics_out["degraded"])
                else:
                    output_tensor = physics_out["degraded"]
                degraded = _crop_to_size(torch.clamp(output_tensor, -1, 1), original_size)
                save_image(degraded.add(1).div(2), sample_dir / f"{image_path.stem}_{key}.png")

                metadata[key] = {
                    "t": float(t_map[key].item()),
                    "readout_sigma_max": float(model.physics_forward.readout_sigma_max),
                    "scatter_strength_max": float(model.physics_forward.scatter_strength_max),
                    "sample_noise": bool(run_sample_noise),
                    "output_mode": args.output_mode,
                    "input_mode": "grayscale_replicated_rgb",
                    "sigma_geo": float(physics_out["sigma_geo"].mean().item()),
                    "sigma_det": float(physics_out["sigma_det"].mean().item()),
                    "readout_sigma": float(physics_out["readout_sigma"].mean().item()),
                    "scatter_strength": float(physics_out["scatter_strength"].mean().item()),
                    "particle_count_effective": float(physics_out["particle_count_effective"].mean().item()),
                }

            (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved z-only degradation inference results to {output_root.resolve()}")


if __name__ == "__main__":
    main()
