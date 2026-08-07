from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

from datasets.neutron_deg import TRAJECTORY_ORDER, build_neutron_index
from utils.common import instantiate_from_config
from utils.interpolation import linear_interpolate, natural_cubic_spline_interpolate
from utils.neutron_schedule import export_t_map_from_state_dict

FLAT_SUFFIX_TO_KEY = {
    "3y": "GT",
    "5k": "LQ_50",
    "3k": "LQ_30",
    "2k": "LQ_20",
    "1k": "LQ_10",
}

PAIR_SUFFIX_TO_KEY = {
    "1k": "LQ_10",
    "2k": "LQ_20",
    "3k": "LQ_30",
    "5k": "LQ_50",
}


def resolve_project_path(project_root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_runtime_configs(train_config_path: str | Path, split: str):
    train_config_path = Path(train_config_path).resolve()
    project_root = train_config_path.parent.parent
    train_cfg = OmegaConf.load(train_config_path)

    model_cfg_path = resolve_project_path(project_root, train_cfg.model.config)
    if model_cfg_path is None:
        raise ValueError("Missing model.config in train config")
    model_cfg = OmegaConf.load(model_cfg_path)

    if split == "train":
        dataset_rel = train_cfg.data.params.train_config
    elif split == "val":
        dataset_rel = train_cfg.data.params.val_config
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset_cfg_path = resolve_project_path(project_root, dataset_rel)
    if dataset_cfg_path is None:
        raise ValueError("Missing dataset config path in train config")
    dataset_cfg = OmegaConf.load(dataset_cfg_path)

    return project_root, train_cfg, model_cfg, dataset_cfg


def load_group_from_dataset_config(
    dataset_cfg,
    project_root: Path,
    prefix: str,
    dataset_root_override: str | None = None,
):
    dataset_params = dataset_cfg.dataset.params
    dataroot = dataset_root_override or dataset_params.dataroot
    dataset_root = resolve_project_path(project_root, dataroot)
    groups = build_neutron_index(
        str(dataset_root),
        hq_subdir=str(dataset_params.get("hq_subdir", "HQ")),
        lq_subdir=str(dataset_params.get("lq_subdir", "LQ")),
    )

    for group in groups:
        if group["prefix"] == prefix:
            return group, dataset_root

    raise FileNotFoundError(f"Prefix not found in dataset index: {prefix}")


def _split_flat_group_name(stem: str) -> tuple[str, str] | None:
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    if suffix not in FLAT_SUFFIX_TO_KEY:
        return None
    return prefix, suffix


def build_groups_from_flat_dir(image_dir: str | Path) -> list[dict]:
    image_dir = Path(image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    grouped_paths: dict[str, dict[str, Path]] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = _split_flat_group_name(path.stem)
        if parsed is None:
            continue
        prefix, suffix = parsed
        grouped_paths.setdefault(prefix, {})[suffix] = path

    groups = []
    required_suffixes = set(FLAT_SUFFIX_TO_KEY.keys())
    for prefix in sorted(grouped_paths.keys()):
        suffix_map = grouped_paths[prefix]
        if set(suffix_map.keys()) != required_suffixes:
            continue
        group = {"prefix": prefix}
        for suffix, key in FLAT_SUFFIX_TO_KEY.items():
            group[key] = suffix_map[suffix]
        groups.append(group)

    if not groups:
        raise FileNotFoundError(f"No complete 3y/5k/3k/2k/1k groups found in {image_dir}")

    return groups


def build_groups_from_hq_lq_dirs(hq_dir: str | Path, lq_dir: str | Path) -> list[dict]:
    hq_dir = Path(hq_dir).resolve()
    lq_dir = Path(lq_dir).resolve()
    if not hq_dir.is_dir():
        raise FileNotFoundError(f"HQ directory not found: {hq_dir}")
    if not lq_dir.is_dir():
        raise FileNotFoundError(f"LQ directory not found: {lq_dir}")

    lq_groups: dict[str, dict[str, Path]] = {}
    for path in sorted(lq_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = _split_flat_group_name(path.stem)
        if parsed is None:
            continue
        prefix, suffix = parsed
        if suffix not in PAIR_SUFFIX_TO_KEY:
            continue
        lq_groups.setdefault(prefix, {})[suffix] = path

    groups = []
    required_suffixes = set(PAIR_SUFFIX_TO_KEY.keys())
    for prefix in sorted(lq_groups.keys()):
        suffix_map = lq_groups[prefix]
        if set(suffix_map.keys()) != required_suffixes:
            continue

        sample_suffix = "1k"
        sample_hq_path = hq_dir / f"{prefix}_{sample_suffix}{suffix_map[sample_suffix].suffix}"
        if not sample_hq_path.exists():
            alt_hq_paths = sorted(hq_dir.glob(f"{prefix}_*"))
            if not alt_hq_paths:
                continue
            sample_hq_path = alt_hq_paths[0]

        group = {
            "prefix": prefix,
            "GT": sample_hq_path,
        }
        for suffix, key in PAIR_SUFFIX_TO_KEY.items():
            group[key] = suffix_map[suffix]
        groups.append(group)

    if not groups:
        raise FileNotFoundError(f"No complete HQ/LQ groups found under HQ={hq_dir} and LQ={lq_dir}")

    return groups


def select_groups(
    *,
    image_dir: str | None,
    hq_dir: str | None,
    lq_dir: str | None,
    dataset_cfg,
    project_root: Path,
    prefix: str | None,
    dataset_root_override: str | None,
    max_groups: int | None,
):
    if hq_dir or lq_dir:
        if not (hq_dir and lq_dir):
            raise ValueError("Both --hq_dir and --lq_dir must be provided together")
        groups = build_groups_from_hq_lq_dirs(hq_dir, lq_dir)
        if prefix:
            groups = [group for group in groups if group["prefix"] == prefix]
            if not groups:
                raise FileNotFoundError(f"Prefix not found in HQ/LQ directories: {prefix}")
        dataset_root = Path(lq_dir).resolve()
    elif image_dir:
        groups = build_groups_from_flat_dir(image_dir)
        if prefix:
            groups = [group for group in groups if group["prefix"] == prefix]
            if not groups:
                raise FileNotFoundError(f"Prefix not found in flat image directory: {prefix}")
        dataset_root = Path(image_dir).resolve()
    else:
        if not prefix:
            raise ValueError("When --image_dir is not provided, --prefix is required")
        group, dataset_root = load_group_from_dataset_config(
            dataset_cfg,
            project_root=project_root,
            prefix=prefix,
            dataset_root_override=dataset_root_override,
        )
        groups = [group]

    if max_groups is not None:
        groups = groups[: max_groups]

    return groups, dataset_root


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def pad_to_multiple(x: torch.Tensor, multiple: int = 2):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def crop_to_size(x: torch.Tensor, size: tuple[int, int]):
    height, width = size
    return x[..., :height, :width]


def parse_alpha_list(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        alpha = float(token)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Alpha must be in (0, 1): {alpha}")
        values.append(alpha)
    if not values:
        raise ValueError("No valid alphas were provided")
    return values


def load_stage1_modules(
    model_cfg,
    project_root: Path,
    device: torch.device,
    checkpoint_override: str | None = None,
    require_physics: bool = False,
):
    ae_config = model_cfg.params.ae_config
    physics_config = model_cfg.params.physics_config if "physics_config" in model_cfg.params else None

    checkpoint_path = checkpoint_override or ae_config.checkpoint
    checkpoint_path = resolve_project_path(project_root, checkpoint_path)
    if checkpoint_path is None:
        raise ValueError("Unable to resolve stage-1 checkpoint path")

    autoencoder = instantiate_from_config(ae_config).to(device)
    physics_forward = instantiate_from_config(physics_config).to(device) if physics_config else None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_ae_keys = [
        "content_encoder",
        "global_degradation_encoder",
        "local_degradation_encoder",
        "degradation_compressor",
        "decoder",
    ]
    missing_ae_keys = [key for key in required_ae_keys if key not in checkpoint]
    if missing_ae_keys:
        raise ValueError(
            "The latent interpolation scripts require a rewritten stage-1 checkpoint. "
            f"Missing keys: {missing_ae_keys}"
        )

    autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
    autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
    autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
    autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
    autoencoder.decoder.load_state_dict(checkpoint["decoder"])
    autoencoder.eval()

    if require_physics:
        if physics_forward is None:
            raise ValueError("physics_config is missing in the model config")
        if "physics_forward" not in checkpoint:
            raise ValueError("physics_forward weights are missing in the stage-1 checkpoint")
        physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)
        physics_forward.eval()

    t_map = export_t_map_from_state_dict(checkpoint.get("t_params"), device=device, dtype=torch.float32)

    return autoencoder, physics_forward, checkpoint, checkpoint_path, t_map


def load_sample_tensors(group: dict, device: torch.device):
    sample = {}
    original_size = None
    for key in TRAJECTORY_ORDER:
        tensor = _load_rgb_tensor(group[key]).to(device)
        padded, size = pad_to_multiple(tensor, multiple=2)
        sample[key] = padded
        if original_size is None:
            original_size = size
    sample["original_size"] = original_size
    sample["prefix"] = group["prefix"]
    return sample


def build_control_points(autoencoder, sample: dict):
    content_dict = autoencoder.encode_content(sample["GT"])
    feature_hr = content_dict["features"]
    control_points = [content_dict["z_c"]]

    for key in TRAJECTORY_ORDER[1:]:
        degradation_dict = autoencoder.encode_degradation(sample[key])
        control_points.append(degradation_dict["z_t"])

    return feature_hr, control_points


def build_control_positions(t_map: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([t_map[key] for key in TRAJECTORY_ORDER], dim=0)


def compute_query_times(
    t_map: dict[str, torch.Tensor],
    start_key: str,
    end_key: str,
    alphas: Iterable[float],
) -> list[tuple[float, torch.Tensor]]:
    t_start = t_map[start_key]
    t_end = t_map[end_key]
    if not float(t_end.item()) > float(t_start.item()):
        raise ValueError(f"Expected {start_key} to come before {end_key} in t-space")

    queries = []
    for alpha in alphas:
        alpha_tensor = torch.tensor(float(alpha), device=t_start.device, dtype=t_start.dtype)
        t_query = t_start + alpha_tensor * (t_end - t_start)
        queries.append((float(alpha), t_query))
    return queries


def interpolate_latent(
    control_points: list[torch.Tensor],
    control_positions: torch.Tensor,
    t_query: torch.Tensor,
    method: str,
) -> torch.Tensor:
    control_tensor = torch.stack(control_points, dim=0)
    interp_factor = t_query.view(1)

    if method == "linear":
        interp, _, _, _ = linear_interpolate(control_tensor, interp_factor, control_positions)
    elif method == "cubic_spline":
        interp, _, _, _ = natural_cubic_spline_interpolate(control_tensor, interp_factor, control_positions)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    return interp


def save_reference_images(sample: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(TRAJECTORY_ORDER):
        save_image(crop_to_size(sample[key], sample["original_size"]).add(1).div(2), output_dir / f"{index:02d}_{key}.png")


def save_metadata(
    output_dir: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    start_key: str,
    end_key: str,
    method: str,
    alphas: list[float],
    t_map: dict[str, torch.Tensor],
):
    lines = [
        f"checkpoint={checkpoint_path}",
        f"dataset_root={dataset_root}",
        f"segment={start_key}->{end_key}",
        f"interp_method={method}",
        f"alphas={','.join(str(alpha) for alpha in alphas)}",
    ]
    for key in TRAJECTORY_ORDER:
        lines.append(f"t_{key}={float(t_map[key].item()):.8f}")
    (output_dir / "metadata.txt").write_text("\n".join(lines), encoding="utf-8")
