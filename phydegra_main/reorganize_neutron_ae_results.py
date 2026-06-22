#ae阶段结果整理
from argparse import ArgumentParser
from pathlib import Path
import shutil


ORIGINAL_MAP = {
    "00_GT.png": ("original", "3y"),
    "02_LQ_50.png": ("original", "5k"),
    "04_LQ_30.png": ("original", "3k"),
    "06_LQ_20.png": ("original", "2k"),
    "08_LQ_10.png": ("original", "1k"),
}

RESULT_MAP = {
    "01_RECON_GT.png": ("result", "3y"),
    "03_RECON_LQ_50.png": ("result", "5k"),
    "05_RECON_LQ_30.png": ("result", "3k"),
    "07_RECON_LQ_20.png": ("result", "2k"),
    "09_RECON_LQ_10.png": ("result", "1k"),
}

FILE_MAP = {**ORIGINAL_MAP, **RESULT_MAP}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Directory containing per-group AE outputs, e.g. result/neutron_ae_524_test2_all",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory to save flattened outputs, e.g. result/neutron_ae_524_test2_all/pair_not_group",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    return parser.parse_args()


def canonical_prefix(group_name: str) -> str:
    if group_name.endswith("_3y"):
        return group_name[: -len("_3y")]
    return group_name


def build_target_name(group_name: str, suffix: str) -> str:
    prefix = canonical_prefix(group_name)
    return f"{prefix}_{suffix}.png"


def ensure_dirs(output_root: Path):
    (output_root / "original").mkdir(parents=True, exist_ok=True)
    (output_root / "result").mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    ensure_dirs(output_root)

    op = shutil.copy2 if args.copy else shutil.move
    processed = 0

    for group_dir in sorted(input_root.iterdir()):
        if not group_dir.is_dir():
            continue
        if group_dir.name == output_root.name:
            continue

        for src_name, (subset, suffix) in FILE_MAP.items():
            src_path = group_dir / src_name
            if not src_path.is_file():
                continue

            dst_name = build_target_name(group_dir.name, suffix)
            dst_path = output_root / subset / dst_name
            op(str(src_path), str(dst_path))
            processed += 1

    print(f"Processed {processed} files into {output_root}")


if __name__ == "__main__":
    main()