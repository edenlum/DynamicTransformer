import wandb
import hashlib
from omegaconf import OmegaConf
import logging


def check_existing_run(entity_name, project_name, cfg, irrelevant_keys=[]):
    api = wandb.Api()
    cfg_hash = get_config_hash(filter_config(cfg, irrelevant_keys=irrelevant_keys))
    try:
        runs = api.runs(f"{entity_name}/{project_name}")
        for run in runs:
            logging.info(f"Current cfg: {cfg} \nWandB cfg: {filter_config(run.config, irrelevant_keys)}")
            if cfg_hash == get_config_hash(filter_config(run.config, irrelevant_keys)):
                return True
    except wandb.errors.CommError as e:
        print(f"Error communicating with wandb API: {e}")
        return False  # Proceeding despite the error
    return False

def get_config_hash(cfg):
    # Convert the config to a JSON string with consistent key ordering
    config_str = str(OmegaConf.to_yaml(cfg))
    # Compute the MD5 hash of the config string
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    return config_hash

def filter_config(config, irrelevant_keys):
    return {k: v for k, v in config.items() if k not in irrelevant_keys}
