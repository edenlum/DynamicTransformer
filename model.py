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
        self.model_name = self.hparams.model_name
        self.skip_layer = self.hparams.skip_layer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, pad_token_id=self.tokenizer.eos_token_id)

        if self.skip_layer is not None:
            self.modify_model(self.skip_layer)


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
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        loss = outputs.loss
        self.log(
            'test_loss', 
            loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True
        )
        return loss


    def on_test_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['test_loss']
        perplexity = torch.exp(avg_loss)
        self.log('test_perplexity', perplexity, prog_bar=True)
        print(f'Test Perplexity: {perplexity.item():.2f}')

    def configure_optimizers(self):
        # Not needed since we're only evaluating
        return None
