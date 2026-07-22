from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import shutil


LQ_TO_SUFFIX = {
    "LQ_10": "1k",
    "LQ_20": "2k",
    "LQ_30": "3k",
    "LQ_50": "5k",
}

ANCHOR_SUFFIXES = ("_1k", "_2k", "_3k", "_5k")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Directory containing per-sample z-only inference folders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory to save flattened renamed images.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default is copy.",
    )
    return parser.parse_args()


def canonical_prefix(group_name: str) -> str:
    for suffix in ANCHOR_SUFFIXES:
        if group_name.endswith(suffix):
            return group_name[: -len(suffix)]
    return group_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if sha256(src) == sha256(dst):
            print(f"Skip duplicate (same content): {dst.name}")
            return
        raise FileExistsError(
            f"Target already exists with different content: {dst}\n"
            f"Source: {src}"
        )

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing = 0

    for group_dir in sorted(input_root.iterdir()):
        if not group_dir.is_dir():
            continue
        if group_dir.resolve() == output_root.resolve():
            continue

        prefix = canonical_prefix(group_dir.name)

        for lq_key, particle_suffix in LQ_TO_SUFFIX.items():
            src_name = f"{group_dir.name}_{lq_key}.png"
            src_path = group_dir / src_name

            if not src_path.is_file():
                print(f"Missing file: {src_path}")
                missing += 1
                continue

            dst_name = f"{prefix}_{particle_suffix}.png"
            dst_path = output_root / dst_name
            copy_or_move(src_path, dst_path, move=args.move)
            processed += 1

    print(f"Processed {processed} images into {output_root}")
    if missing > 0:
        print(f"Missing expected files: {missing}")


if __name__ == "__main__":
    main()
