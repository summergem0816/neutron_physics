import random
from pathlib import Path

import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import augment_img
from utils.neutron_schedule import build_t_map


PARTICLE_SUFFIX_MAP = {
    "1k": 10,
    "2k": 20,
    "3k": 30,
    "5k": 50,
}

T_MAP = build_t_map()

LQ_KEY_TO_SUFFIX = {
    "LQ_10": "1k",
    "LQ_20": "2k",
    "LQ_30": "3k",
    "LQ_50": "5k",
}

PAIR_ORDER = ["LQ_50", "LQ_30", "LQ_20", "LQ_10"]
TRAJECTORY_ORDER = ["GT", "LQ_50", "LQ_30", "LQ_20", "LQ_10"]


def _is_rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _split_prefix_and_suffix(stem: str) -> tuple[str, str] | None:
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    if suffix not in PARTICLE_SUFFIX_MAP:
        return None
    return prefix, suffix


def _load_rgb(path: Path) -> np.ndarray:
    gray = np.array(Image.open(path).convert("L"))
    return np.repeat(gray[:, :, None], 3, axis=2)


def _to_tensor_image(img: np.ndarray) -> np.ndarray:
    return img.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0


def _crop_all(images: dict[str, np.ndarray], patch_size: int, test: bool) -> dict[str, np.ndarray]:
    ref = images["GT"]
    height, width, _ = ref.shape
    patch_h = min(patch_size, height)
    patch_w = min(patch_size, width)
    if test:
        top = max(0, (height - patch_h) // 2)
        left = max(0, (width - patch_w) // 2)
    else:
        top = random.randint(0, max(0, height - patch_h))
        left = random.randint(0, max(0, width - patch_w))

    return {
        key: img[top:top + patch_h, left:left + patch_w, :]
        for key, img in images.items()
    }


def _augment_all(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mode = random.randint(0, 7)
    return {key: augment_img(img, mode) for key, img in images.items()}


def build_neutron_index(
    dataroot: str,
    hq_subdir: str = "HQ",
    lq_subdir: str = "LQ",
    require_complete_group: bool = True,
) -> list[dict]:
    root = Path(dataroot)
    hq_dir = root / hq_subdir
    lq_dir = root / lq_subdir

    if not hq_dir.is_dir():
        raise FileNotFoundError(f"HQ directory not found: {hq_dir}")
    if not lq_dir.is_dir():
        raise FileNotFoundError(f"LQ directory not found: {lq_dir}")

    groups: list[dict] = []

    # Only use HQ *_1k files as the unique anchor for each group.
    for hq_path in sorted(hq_dir.iterdir()):
        if not hq_path.is_file():
            continue
        parsed = _split_prefix_and_suffix(hq_path.stem)
        if parsed is None:
            continue
        prefix, suffix = parsed
        if suffix != "1k":
            continue

        lq_paths = {}
        missing = []
        for key, lq_suffix in LQ_KEY_TO_SUFFIX.items():
            candidate = lq_dir / f"{prefix}_{lq_suffix}{hq_path.suffix}"
            if candidate.exists():
                lq_paths[key] = candidate
            else:
                missing.append(str(candidate))

        if require_complete_group and missing:
            continue
        if len(lq_paths) != len(LQ_KEY_TO_SUFFIX):
            continue

        groups.append(
            {
                "prefix": prefix,
                "GT": hq_path,
                "LQ_10": lq_paths["LQ_10"],
                "LQ_20": lq_paths["LQ_20"],
                "LQ_30": lq_paths["LQ_30"],
                "LQ_50": lq_paths["LQ_50"],
                "t_values": dict(T_MAP),
                "particle_counts": {
                    "GT": 300,
                    "LQ_10": 10,
                    "LQ_20": 20,
                    "LQ_30": 30,
                    "LQ_50": 50,
                },
            }
        )

    if not groups:
        raise FileNotFoundError(f"No complete neutron groups found under {root}")

    if _is_rank_zero():
        print(f"Loaded {len(groups)} neutron groups from {root}")

    return groups


class NeutronPairDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        augmentation: bool,
        test: bool,
        patch_size: int = None,
        preload: bool = False,
        hq_subdir: str = "HQ",
        lq_subdir: str = "LQ",
        object_map_path: str = None,
        view_map_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.test = test
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload
        self.groups = build_neutron_index(dataroot, hq_subdir=hq_subdir, lq_subdir=lq_subdir)

        self.samples: list[dict] = []
        for group in self.groups:
            for lq_key in PAIR_ORDER:
                self.samples.append(
                    {
                        "prefix": group["prefix"],
                        "GT": group["GT"],
                        "LQ": group[lq_key],
                        "lq_key": lq_key,
                        "t": group["t_values"][lq_key],
                        "particle_count": group["particle_counts"][lq_key],
                    }
                )

        if self.preload:
            self.imgs: dict[str, np.ndarray] = {}
            for sample in self.samples:
                for path in [sample["GT"], sample["LQ"]]:
                    key = str(path)
                    if key not in self.imgs:
                        self.imgs[key] = _load_rgb(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        gt = self.imgs[str(sample["GT"])] if self.preload else _load_rgb(sample["GT"])
        lq = self.imgs[str(sample["LQ"])] if self.preload else _load_rgb(sample["LQ"])

        images = {"GT": gt, "LQ": lq}

        if self.patch_size is not None:
            images = _crop_all(images, self.patch_size, self.test)
        if not self.test and self.augmentation:
            images = _augment_all(images)

        item = {
            "GT": _to_tensor_image(images["GT"]),
            "LQ": _to_tensor_image(images["LQ"]),
            "t": np.float32(sample["t"]),
            "particle_count": np.int64(sample["particle_count"]),
            "prefix": sample["prefix"],
            "lq_key": sample["lq_key"],
        }
        if self.test:
            item["img_idx"] = index
        return item


class NeutronTrajectoryDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        augmentation: bool,
        test: bool,
        patch_size: int = None,
        preload: bool = False,
        hq_subdir: str = "HQ",
        lq_subdir: str = "LQ",
        object_map_path: str = None,
        view_map_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.test = test
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.preload = preload
        self.groups = build_neutron_index(dataroot, hq_subdir=hq_subdir, lq_subdir=lq_subdir)

        if self.preload:
            self.imgs: dict[str, np.ndarray] = {}
            for group in self.groups:
                for key in TRAJECTORY_ORDER:
                    path = group[key]
                    str_path = str(path)
                    if str_path not in self.imgs:
                        self.imgs[str_path] = _load_rgb(path)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        group = self.groups[index]

        images = {}
        for key in TRAJECTORY_ORDER:
            path = group[key]
            images[key] = self.imgs[str(path)] if self.preload else _load_rgb(path)

        if self.patch_size is not None:
            images = _crop_all(images, self.patch_size, self.test)
        if not self.test and self.augmentation:
            images = _augment_all(images)

        item = {
            key: _to_tensor_image(images[key])
            for key in TRAJECTORY_ORDER
        }
        item["t_values"] = np.asarray(
            [group["t_values"][key] for key in TRAJECTORY_ORDER],
            dtype=np.float32,
        )
        item["particle_counts"] = np.asarray(
            [group["particle_counts"][key] for key in TRAJECTORY_ORDER],
            dtype=np.int64,
        )
        item["prefix"] = group["prefix"]

        if self.test:
            item["img_idx"] = index

        return item
