# train_router.py

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_module import WikiTextDataModule
from gpt2_router import GPT2WithRouter
import torch
import wandb
from utils import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path='configs', config_name='train_router', version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision(cfg.torch.precision)

    irrelevant_keys = ['hydra', 'experiment', 'trainer']
    if check_existing_run(cfg.experiment.entity, cfg.experiment.project, cfg, irrelevant_keys=irrelevant_keys):
        print("An experiment with this configuration has already been run.")
        return  # Exit or proceed based on your preference

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        name=f"{cfg.experiment.name}_model={cfg.model.name}_block_size={cfg.model.block_size}",
        project=cfg.experiment.project,
        entity=cfg.experiment.entity,
        config=filter_config(cfg, irrelevant_keys),
        reinit=True
    ) if cfg.trainer.logger else None

    # Initialize trainer
    trainer = Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        max_epochs=cfg.trainer.max_epochs
    )

    # Initialize data module for training
    data_module = WikiTextDataModule(
        tokenizer_name=cfg.model.name,
        block_size=cfg.data.block_size,
        stride=cfg.data.stride,
        split='train',
        batch_size=cfg.data.batch_size
    )

    # Initialize model
    model = GPT2WithRouter(
        model_name=cfg.model.name,
        block_size=cfg.model.block_size,
        iters=cfg.model.iters,
        learning_rate=cfg.model.learning_rate
    )

    # Run training
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    # use_free_gpus()
    main()