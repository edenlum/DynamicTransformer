# test_router.py

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_module import WikiTextDataModule
from gpt2_router import GPT2WithRouter
import torch
import wandb
from utils import *

@hydra.main(config_path='configs', config_name='test_router', version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision(cfg.torch.precision)

    # Initialize wandb logger (if logging is enabled)
    wandb_logger = WandbLogger(
        name=f"{cfg.experiment.name}_model={cfg.model.name}_block_size={cfg.model.block_size}_test",
        project=cfg.experiment.project,
        entity=cfg.experiment.entity,
        config=filter_config(cfg, ['hydra', 'experiment', 'trainer']),
        reinit=True
    ) if cfg.trainer.logger else None

    # Initialize trainer for testing
    trainer = Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        enable_checkpointing=False,  # No checkpointing required for testing
        max_epochs=1  # Only one epoch for testing
    )

    # Initialize data module for testing
    data_module = WikiTextDataModule(
        tokenizer_name=cfg.model.name,
        block_size=cfg.data.block_size,
        stride=cfg.data.stride,
        split='test',
        batch_size=cfg.data.batch_size
    )

    # Load the model from the checkpoint file
    model = GPT2WithRouter.load_from_checkpoint(
        cfg.model.checkpoint_path,
        model_name=cfg.model.name,
        block_size=cfg.model.block_size
    )

    # Run testing
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    use_free_gpus()
    main()
