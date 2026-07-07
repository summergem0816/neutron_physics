from argparse import ArgumentParser
from pathlib import Path

import torch
from torchvision.utils import save_image

from interpolate_neutron_latent_common import (
    build_control_points,
    build_control_positions,
    compute_query_times,
    crop_to_size,
    load_runtime_configs,
    load_sample_tensors,
    load_stage1_modules,
    parse_alpha_list,
    interpolate_latent,
    select_groups,
    save_metadata,
    save_reference_images,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/train_neutron_lit_rf.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--prefix", type=str, default=None, help="Optional single-group filter, e.g. z259_on")
    parser.add_argument("--image_dir", type=str, default=None, help="Flat directory containing grouped *_1k/*_2k/*_3k/*_5k/*_3y files")
    parser.add_argument("--hq_dir", type=str, default=None, help="HQ directory containing duplicated 3y images named like *_1k/*_2k/*_3k/*_5k")
    parser.add_argument("--lq_dir", type=str, default=None, help="LQ directory containing *_1k/*_2k/*_3k/*_5k")
    parser.add_argument("--max_groups", type=int, default=500)
    parser.add_argument("--start_key", type=str, default="GT", choices=["GT", "LQ_50", "LQ_30", "LQ_20"])
    parser.add_argument("--end_key", type=str, default="LQ_50", choices=["LQ_50", "LQ_30", "LQ_20", "LQ_10"])
    parser.add_argument("--alphas", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--interp_method", type=str, default="cubic_spline", choices=["linear", "cubic_spline"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    alphas = parse_alpha_list(args.alphas)

    project_root, _, model_cfg, dataset_cfg = load_runtime_configs(args.train_config, split=args.split)
    groups, dataset_root = select_groups(
        image_dir=args.image_dir,
        hq_dir=args.hq_dir,
        lq_dir=args.lq_dir,
        dataset_cfg=dataset_cfg,
        project_root=project_root,
        prefix=args.prefix,
        dataset_root_override=args.dataset_root,
        max_groups=args.max_groups,
    )
    autoencoder, _, _, checkpoint_path, t_map = load_stage1_modules(
        model_cfg,
        project_root=project_root,
        device=device,
        checkpoint_override=args.ae_checkpoint,
        require_physics=False,
    )

    control_positions = build_control_positions(t_map)
    query_times = compute_query_times(t_map, args.start_key, args.end_key, alphas)

    with torch.no_grad():
        for group in groups:
            sample = load_sample_tensors(group, device=device)
            feature_hr, control_points = build_control_points(autoencoder, sample)

            output_dir = Path(args.output).resolve() / group["prefix"]
            output_dir.mkdir(parents=True, exist_ok=True)
            save_reference_images(sample, output_dir)
            save_metadata(
                output_dir,
                checkpoint_path=checkpoint_path,
                dataset_root=dataset_root,
                start_key=args.start_key,
                end_key=args.end_key,
                method=args.interp_method,
                alphas=alphas,
                t_map=t_map,
            )

            for index, (alpha, t_query) in enumerate(query_times, start=1):
                pred_z = interpolate_latent(
                    control_points=control_points,
                    control_positions=control_positions,
                    t_query=t_query,
                    method=args.interp_method,
                )
                pred_img = autoencoder.reconstruct_from_content_degradation(pred_z, feature_hr)
                pred_img = torch.clamp(pred_img, -1, 1)
                pred_img = crop_to_size(pred_img, sample["original_size"])
                save_image(
                    pred_img.add(1).div(2),
                    output_dir / f"interp_decoder_{args.start_key}_to_{args.end_key}_a{int(round(alpha * 100)):02d}.png",
                )

    print(f"Saved decoder interpolation results to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
