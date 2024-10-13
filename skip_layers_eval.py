import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from data_module import WikiTextDataModule
from model import GPT2LightningModule

@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    # Update experiment name based on skip_layer
    cfg.experiment.name = f'skip_layer_{cfg.model.skip_layer}'
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb logger if logging is enabled
    if cfg.trainer.logger:
        wandb_logger = WandbLogger(
            name=cfg.experiment.name,
            project=cfg.experiment.project
        )
    else:
        wandb_logger = None

    # Initialize trainer
    trainer = Trainer(
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        enable_checkpointing=cfg.trainer.enable_checkpointing
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize data module
    data_module = WikiTextDataModule(
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length
    )

    # Initialize model
    model = GPT2LightningModule(
        model_name=cfg.model.name,
        skip_layer=cfg.model.skip_layer
    )

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
