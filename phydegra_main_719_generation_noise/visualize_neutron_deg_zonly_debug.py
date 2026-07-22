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
LEVEL_TO_SUFFIX = {
    "LQ_10": "1k",
    "LQ_20": "2k",
    "LQ_30": "3k",
    "LQ_50": "5k",
}
SUFFIX_TO_LEVEL = {value: key for key, value in LEVEL_TO_SUFFIX.items()}
PARTICLE_SUFFIXES = set(SUFFIX_TO_LEVEL)


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
    anchors = []
    for path in sorted(hq_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = _split_prefix_and_suffix(path.stem)
        if parsed is None:
            continue
        _, suffix = parsed
        if suffix == "1k":
            anchors.append(path)
    return anchors


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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_unit_range(x: torch.Tensor, per_channel: bool = True) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 2:
        x = x.unsqueeze(0)

    if per_channel:
        flat = x.flatten(1)
        min_v = flat.min(dim=1).values[:, None, None]
        max_v = flat.max(dim=1).values[:, None, None]
    else:
        min_v = x.min()
        max_v = x.max()
    return (x - min_v) / (max_v - min_v).clamp_min(1e-8)


def _save_feature_grid(x: torch.Tensor, path: Path, max_channels: int = 16) -> None:
    vis = _to_unit_range(x, per_channel=True)
    vis = vis[:max_channels]
    nrow = min(4, max(1, vis.shape[0]))
    save_image(vis, path, nrow=nrow)


def _save_feature_summaries(x: torch.Tensor, output_dir: Path, prefix: str) -> None:
    tensor = x.detach()
    mean_map = tensor.mean(dim=1, keepdim=True)
    abs_mean_map = tensor.abs().mean(dim=1, keepdim=True)
    _save_feature_grid(mean_map, output_dir / f"{prefix}_mean_map.png", max_channels=1)
    _save_feature_grid(abs_mean_map, output_dir / f"{prefix}_abs_mean_map.png", max_channels=1)


def _save_image_minus1_1(x: torch.Tensor, path: Path, size: tuple[int, int] | None = None) -> None:
    if size is not None:
        x = _crop_to_size(x, size)
    save_image(x.detach().clamp(-1, 1).add(1).div(2).cpu(), path)


def _save_image_0_1(x: torch.Tensor, path: Path, size: tuple[int, int] | None = None) -> None:
    if size is not None:
        x = _crop_to_size(x, size)
    save_image(x.detach().clamp(0, 1).cpu(), path)


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(x.abs().mean().item()),
    }


def _z_alignment_stats(pred_z: torch.Tensor, target_z: torch.Tensor) -> dict[str, float]:
    pred_flat = pred_z.flatten(1)
    target_flat = target_z.flatten(1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1)
    return {
        "l1": float(F.l1_loss(pred_z, target_z).item()),
        "mse": float(F.mse_loss(pred_z, target_z).item()),
        "cosine": float(cosine.mean().item()),
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/train_neutron_lit_deg_zonly.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hq_dir", type=str, required=True)
    parser.add_argument("--lq_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_noise", action="store_true", default=True)
    parser.add_argument("--no_sample_noise", action="store_false", dest="sample_noise")
    parser.add_argument("--readout_sigma_max", type=float, default=None)
    parser.add_argument("--save_raw_tensors", action="store_true")
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
    lq_dir = Path(args.lq_dir)
    image_paths = _collect_anchor_image_paths(hq_dir)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    output_root = _ensure_dir(Path(args.output))
    diagnostics_lines = [
        "sample\tlevel\tsuffix\tt\tpred_z_mean\tpred_z_std\ttarget_z_mean\ttarget_z_std\t"
        "z_l1\tz_cosine\treadout_map_mean\treadout_map_std\tscatter_map_mean\tscatter_map_std\t"
        "readout_sigma\tscatter_strength\tmean_to_sample_abs\tmean_to_sample_std"
    ]

    print(f"Using {len(image_paths)} HQ anchors from {hq_dir}; each anchor is matched to 1k/2k/3k/5k LQ images.")
    print(f"Saving stochastic outputs: {args.sample_noise}")

    with torch.no_grad():
        for image_path in image_paths:
            parsed = _split_prefix_and_suffix(image_path.stem)
            if parsed is None:
                continue
            sample_prefix, _ = parsed

            clean = _load_rgb_tensor(image_path).to(device)
            clean_padded, original_size = _pad_to_multiple(clean, multiple=2)

            content_dict = model.autoencoder.encode_content(clean_padded)
            z_content = content_dict["z_c"]

            sample_dir = _ensure_dir(output_root / sample_prefix)
            _save_image_minus1_1(clean, sample_dir / f"{sample_prefix}_HQ_anchor_1k.png")

            latent_dir = _ensure_dir(sample_dir / "latents")
            physics_dir = _ensure_dir(sample_dir / "physics")
            maps_dir = _ensure_dir(sample_dir / "condition_maps")
            final_dir = _ensure_dir(sample_dir / "final_outputs")

            _save_feature_grid(z_content, latent_dir / "z_content_grid.png")
            _save_feature_summaries(z_content, latent_dir, "z_content")

            metadata: dict[str, object] = {
                "hq_anchor": str(image_path),
                "sample_prefix": sample_prefix,
                "sample_noise_saved": bool(args.sample_noise),
                "levels": {},
            }

            for key in LEVEL_KEYS:
                suffix = LEVEL_TO_SUFFIX[key]
                lq_path = lq_dir / f"{sample_prefix}_{suffix}.png"
                if not lq_path.exists():
                    print(f"Missing LQ image for {sample_prefix} {suffix}: {lq_path}")
                    continue

                real_lq = _load_rgb_tensor(lq_path).to(device)
                real_lq_padded, _ = _pad_to_multiple(real_lq, multiple=2)
                _save_image_minus1_1(real_lq, final_dir / f"{sample_prefix}_real_{suffix}.png")

                degradation = model.autoencoder.encode_degradation(real_lq_padded)
                target_z = degradation["z_t"]
                local_code = degradation["local_code"]
                global_code = degradation["global_code"]

                t = t_map[key].expand(clean.shape[0]).to(device=device, dtype=clean.dtype)
                pred_z = model.predictor(z_content=z_content, clean=clean_padded, t=t)["pred_z"]

                mean_out = model.physics_forward(
                    clean=clean_padded,
                    t=t,
                    degradation_latent=pred_z,
                    sample_noise=False,
                )
                sample_out = None
                if args.sample_noise:
                    sample_out = model.physics_forward(
                        clean=clean_padded,
                        t=t,
                        degradation_latent=pred_z,
                        sample_noise=True,
                    )

                level_latent_dir = _ensure_dir(latent_dir / suffix)
                _save_feature_grid(target_z, level_latent_dir / f"target_z_t_{suffix}_grid.png")
                _save_feature_grid(pred_z, level_latent_dir / f"pred_z_t_{suffix}_grid.png")
                _save_feature_grid((pred_z - target_z).abs(), level_latent_dir / f"abs_diff_pred_target_z_{suffix}_grid.png")
                _save_feature_grid(local_code, level_latent_dir / f"real_local_code_{suffix}_grid.png")
                _save_feature_summaries(target_z, level_latent_dir, f"target_z_t_{suffix}")
                _save_feature_summaries(pred_z, level_latent_dir, f"pred_z_t_{suffix}")
                _save_feature_summaries((pred_z - target_z).abs(), level_latent_dir, f"abs_diff_pred_target_z_{suffix}")
                np.save(level_latent_dir / f"real_global_code_{suffix}.npy", global_code.detach().cpu().numpy())

                if args.save_raw_tensors:
                    torch.save(
                        {
                            "z_content": z_content.detach().cpu(),
                            "target_z": target_z.detach().cpu(),
                            "pred_z": pred_z.detach().cpu(),
                            "local_code": local_code.detach().cpu(),
                            "global_code": global_code.detach().cpu(),
                        },
                        level_latent_dir / f"raw_latents_{suffix}.pt",
                    )

                _save_feature_grid(mean_out["readout_map"], maps_dir / f"readout_map_{suffix}.png", max_channels=1)
                _save_feature_grid(mean_out["scatter_map"], maps_dir / f"scatter_map_{suffix}.png", max_channels=1)
                _save_feature_grid(mean_out["interaction_gate"], maps_dir / f"interaction_gate_{suffix}.png", max_channels=1)

                _save_image_0_1(mean_out["source"], physics_dir / f"source_{suffix}.png", original_size)
                _save_image_0_1(mean_out["geo_blur"], physics_dir / f"geo_blur_{suffix}.png", original_size)
                _save_image_0_1(mean_out["det_blur"], physics_dir / f"det_blur_{suffix}.png", original_size)
                _save_image_0_1(mean_out["blur_interaction"], physics_dir / f"blur_interaction_{suffix}.png", original_size)
                _save_image_0_1(mean_out["scatter"], physics_dir / f"scatter_{suffix}.png", original_size)
                _save_image_0_1(mean_out["pre_poisson"], physics_dir / f"pre_poisson_{suffix}.png", original_size)

                mean_degraded = mean_out["degraded"]
                _save_image_minus1_1(mean_degraded, final_dir / f"{sample_prefix}_pred_{suffix}_mean_no_sample_noise.png", original_size)

                sample_delta_abs = 0.0
                sample_delta_std = 0.0
                if sample_out is not None:
                    sample_degraded = sample_out["degraded"]
                    _save_image_minus1_1(sample_degraded, final_dir / f"{sample_prefix}_pred_{suffix}_sample_noise.png", original_size)
                    delta = _crop_to_size(sample_degraded - mean_degraded, original_size)
                    _save_feature_grid(delta, final_dir / f"{sample_prefix}_noise_residual_{suffix}_grid.png")
                    sample_delta_abs = float(delta.abs().mean().item())
                    sample_delta_std = float(delta.std(unbiased=False).item())

                z_align = _z_alignment_stats(pred_z, target_z)
                level_meta = {
                    "real_lq_path": str(lq_path),
                    "t": float(t_map[key].item()),
                    "suffix": suffix,
                    "target_z": _tensor_stats(target_z),
                    "pred_z": _tensor_stats(pred_z),
                    "local_code": _tensor_stats(local_code),
                    "z_alignment": z_align,
                    "readout_map": _tensor_stats(mean_out["readout_map"]),
                    "scatter_map": _tensor_stats(mean_out["scatter_map"]),
                    "interaction_gate": _tensor_stats(mean_out["interaction_gate"]),
                    "readout_sigma": float(mean_out["readout_sigma"].mean().item()),
                    "scatter_strength": float(mean_out["scatter_strength"].mean().item()),
                    "particle_count_effective": float(mean_out["particle_count_effective"].mean().item()),
                    "mean_to_sample_abs": sample_delta_abs,
                    "mean_to_sample_std": sample_delta_std,
                }
                metadata["levels"][key] = level_meta

                diagnostics_lines.append(
                    f"{sample_prefix}\t{key}\t{suffix}\t{level_meta['t']:.6g}\t"
                    f"{level_meta['pred_z']['mean']:.6g}\t{level_meta['pred_z']['std']:.6g}\t"
                    f"{level_meta['target_z']['mean']:.6g}\t{level_meta['target_z']['std']:.6g}\t"
                    f"{z_align['l1']:.6g}\t{z_align['cosine']:.6g}\t"
                    f"{level_meta['readout_map']['mean']:.6g}\t{level_meta['readout_map']['std']:.6g}\t"
                    f"{level_meta['scatter_map']['mean']:.6g}\t{level_meta['scatter_map']['std']:.6g}\t"
                    f"{level_meta['readout_sigma']:.6g}\t{level_meta['scatter_strength']:.6g}\t"
                    f"{sample_delta_abs:.6g}\t{sample_delta_std:.6g}"
                )

            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (output_root / "linearity_diagnostics.txt").write_text("\n".join(diagnostics_lines) + "\n", encoding="utf-8")
    print(f"Saved debug visualizations to {output_root.resolve()}")
    print(f"Saved diagnostics table to {(output_root / 'linearity_diagnostics.txt').resolve()}")


if __name__ == "__main__":
    main()
