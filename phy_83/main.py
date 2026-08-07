from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from utils.common import get_obj_from_str, instantiate_from_config


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if config.lightning.seed:
        pl.seed_everything(config.lightning.seed, workers=True)
    if config.lightning.get("matmul_precision"):
        torch.set_float32_matmul_precision(config.lightning.get("matmul_precision"))

    data_module = instantiate_from_config(config.data)
    model_config = OmegaConf.load(config.model.config)
    if config.model.get("pl_resume"):
        model = get_obj_from_str(model_config.target).load_from_checkpoint(
            config.model.get("pl_resume"),
            strict=True,
            map_location="cpu",
        )
    else:
        model = instantiate_from_config(model_config)

    if config.model.get("resume"):
        checkpoint_path = config.model.resume
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_type = Path(config.model.config).stem
        if model_type.endswith("rf"):
            model.model.load_state_dict(checkpoint["model"])
            if "t_params" in checkpoint:
                model.t_params.load_export_state_dict(checkpoint["t_params"])
            print(f"[RF] Checkpoint loaded from {checkpoint_path}")
        elif model_type.endswith("ae"):
            if "content_encoder" in checkpoint:
                model.autoencoder.content_encoder.load_state_dict(checkpoint["content_encoder"])
                model.autoencoder.global_degradation_encoder.load_state_dict(checkpoint["global_degradation_encoder"])
                model.autoencoder.local_degradation_encoder.load_state_dict(checkpoint["local_degradation_encoder"])
                model.autoencoder.degradation_compressor.load_state_dict(checkpoint["degradation_compressor"])
            else:
                model.autoencoder.content_encoder.load_state_dict(checkpoint["encoder"])
            model.autoencoder.decoder.load_state_dict(checkpoint["decoder"])
            model.t_params.load_export_state_dict(checkpoint["t_params"])
            if getattr(model, "physics_forward", None) is not None and "physics_forward" in checkpoint:
                model.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)
            print(f"[AE] Checkpoint loaded from {checkpoint_path}")
        elif model_type.endswith("zonly"):
            model.predictor.load_state_dict(checkpoint["predictor"])
            if "t_params" in checkpoint:
                model.t_params.load_export_state_dict(checkpoint["t_params"])
            if getattr(model, "physics_forward", None) is not None and "physics_forward" in checkpoint:
                model.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)
            print(f"[ZOnly] Checkpoint loaded from {checkpoint_path}")
        elif model_type.endswith("bridge"):
            model.predictor.load_state_dict(checkpoint["predictor"])
            if "t_params" in checkpoint:
                model.t_params.load_export_state_dict(checkpoint["t_params"])
            if getattr(model, "physics_forward", None) is not None and "physics_forward" in checkpoint:
                model.physics_forward.load_state_dict(checkpoint["physics_forward"], strict=False)
            print(f"[Bridge] Checkpoint loaded from {checkpoint_path}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    callbacks = []
    if "callbacks" in config.lightning.keys():
        for callback_config in config.lightning.callbacks:
            callbacks.append(instantiate_from_config(callback_config))

    if not args.debug:
        loggers = []
        if "loggers" in config.lightning.keys():
            for logger_config in config.lightning.loggers:
                logger = instantiate_from_config(logger_config)
                loggers.append(logger)
                if isinstance(logger, WandbLogger):
                    code = wandb.Artifact("code", type="code")
                    for path in Path(".").glob("**/*.py"):
                        code.add_file(path, name=str(path))
                    for path in Path(".").glob("**/*.yaml"):
                        code.add_file(path, name=str(path))
                    logger.experiment.log_artifact(code)
    else:
        callbacks, loggers = [], []
        rich_progress_bar = instantiate_from_config(
            {
                "target": "pytorch_lightning.callbacks.RichProgressBar",
                "params": {},
            }
        )
        callbacks.append(rich_progress_bar)
        debug_logger = instantiate_from_config(
            {
                "target": "models.loggers.LocalImageLogger",
                "params": {
                    "save_dir": "./logs/",
                    "name": "LocalImageLogger",
                    "version": "debug",
                },
            }
        )
        loggers.append(debug_logger)
        config.lightning.trainer.val_check_interval = 10000
        data_module.train_config.dataset.params.preload = False
        if isinstance(data_module.val_config, list):
            for vc in data_module.val_config:
                vc.dataset.params.preload = False
        elif data_module.val_config is not None:
            data_module.val_config.dataset.params.preload = False

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **config.lightning.trainer,
        num_sanity_val_steps=0,
    )

    if config.lightning.mode == "fit":
        trainer.fit(model, datamodule=data_module)
    elif config.lightning.mode == "validate":
        trainer.validate(model, datamodule=data_module)
    else:
        raise ValueError(f"Unsupported mode: {config.lightning.mode}")


if __name__ == "__main__":
    main()
