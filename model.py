import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn



class GPT2LightningModule(pl.LightningModule):
    def __init__(self, model_name='gpt2', skip_layer=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.skip_layer = skip_layer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.eos_token_id
        )

        if self.skip_layer is not None:
            print(f"skipping layer {self.skip_layer}")
            self.modify_model(self.skip_layer)
        else:
            print("Not skipping any layers")

        self.nlls = []


    def modify_model(self, layer_to_skip):
        # Modify the model to skip a specific layer
        import copy
        if not isinstance(layers_to_remove, list):
            layers_to_remove = [layers_to_remove]
        modified_model = copy.deepcopy(self.model)
        modified_model.transformer.h = nn.ModuleList(
            [layer for i, layer in enumerate(modified_model.transformer.h) if i not in layers_to_remove]
        )

        self.model = modified_model


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        outputs = self.model(input_ids=input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        self.nlls.append(neg_log_likelihood)

        return neg_log_likelihood

    def on_test_epoch_end(self):
        ppl = torch.exp(torch.stack(self.nlls).mean())
        self.log('test_perplexity', ppl)
        print(f'Test Perplexity: {ppl.item():.2f}')

    def configure_optimizers(self):
        # Not needed since we're only evaluating
        return None
