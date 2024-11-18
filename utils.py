import wandb
import hashlib
from omegaconf import OmegaConf
import logging
import subprocess
import os


def check_existing_run(entity_name, project_name, cfg, irrelevant_keys=[]):
    api = wandb.Api()
    cfg_hash = get_config_hash(filter_config(cfg, irrelevant_keys=irrelevant_keys))
    try:
        runs = api.runs(f"{entity_name}/{project_name}")
        for run in runs:
            logging.debug(f"Current cfg: {cfg} \nWandB cfg: {filter_config(run.config, irrelevant_keys)}")
            if cfg_hash == get_config_hash(filter_config(run.config, irrelevant_keys)):
                return True
    except wandb.errors.CommError as e:
        print(f"Error communicating with wandb API: {e}")
        return False  # Proceeding despite the error
    except ValueError as e:
        print(f"Error: {e}")
        return False
    return False

def get_config_hash(cfg):
    # Convert the config to a JSON string with consistent key ordering
    config_str = str(OmegaConf.to_yaml(cfg))
    # Compute the MD5 hash of the config string
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    return config_hash

def filter_config(config, irrelevant_keys):
    return {k: v for k, v in config.items() if k not in irrelevant_keys}


def get_free_gpus():
    try:
        # Run `nvidia-smi` to get GPU usage details
        result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'])
        running_gpus = [x.strip() for x in result.decode('utf-8').strip().split('\n') if x]

        # Get list of all GPUs
        result_all_gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'])
        all_gpus = [x.strip() for x in result_all_gpus.decode('utf-8').strip().split('\n')]

        # Free GPUs are those without active processes
        free_gpus = [idx for idx, gpu_uuid in enumerate(all_gpus) if gpu_uuid not in running_gpus]

        return free_gpus

    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
        return []

def use_free_gpus():
    # Get the list of free GPUs
    free_gpus = get_free_gpus()
    if not free_gpus:
        raise RuntimeError("No free GPUs are available to run experiments.")

    # Set CUDA_VISIBLE_DEVICES to the free GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
    print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")