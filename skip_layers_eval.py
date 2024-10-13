# main.py

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_module import WikiTextDataModule
from model import GPT2LightningModule

@hydra.main(config_path='configs', config_name='skip_layer')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        name=cfg.experiment.name,
        project=cfg.experiment.project
    ) if cfg.trainer.logger else None

    # Initialize trainer
    trainer = Trainer(
        gpus=cfg.trainer.gpus,
        logger=wandb_logger,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        max_epochs=1,
    )

    # Initialize data module
    data_module = WikiTextDataModule(
        tokenizer_name=cfg.model.name,
        block_size=cfg.data.block_size,
        stride=cfg.data.stride
    )

    # Initialize model
    model = GPT2LightningModule(
        model_name=cfg.model.name,
        skip_layer=cfg.model.skip_layer
    )

    # Run test
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
