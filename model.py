import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT



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
            self.modify_model(self.skip_layer)

        self.nlls = []


    def modify_model(self, layer_to_skip):
        # Modify the model to skip a specific layer
        import types

        def custom_forward(self, input_ids=None, attention_mask=None, **kwargs):
            # Get input embeddings
            input_shape = input_ids.size()
            input_embeds = self.wte(input_ids)
            position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=input_ids.device)
            position_embeds = self.wpe(position_ids)
            hidden_states = input_embeds + position_embeds

            # Apply dropout
            hidden_states = self.drop(hidden_states)

            # Iterate through transformer layers
            for i, block in enumerate(self.h):
                if i == layer_to_skip:
                    continue  # Skip this layer
                outputs = block(hidden_states, attention_mask=attention_mask, **kwargs)
                hidden_states = outputs[0]

            hidden_states = self.ln_f(hidden_states)
            return hidden_states

        # Bind the new forward function to the model's transformer
        self.model.transformer.forward = types.MethodType(custom_forward, self.model.transformer)

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
