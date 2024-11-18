import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import wandb



class GPT2LightningModule(pl.LightningModule):
    def __init__(self, model_name='gpt2', skip_layer=None, swap_layer=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.skip_layer = skip_layer
        self.swap_layer = swap_layer

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Pretrained Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Linear layers that will decide whether to skip each transformer block
        if self.skip_layer is not None:
            print(f"skipping layer {self.skip_layer}")
            self.model_with_skip(self.skip_layer)
        else:
            print("Not skipping any layers")

        print(self.swap_layer)
        if self.swap_layer is not None:
            print(f"swapping layers {self.swap_layer} and {self.swap_layer + 1}")
            self.model_with_swap()
        else:
            print("Not swapping any layers")

        self.per_token_losses = torch.tensor([])
        self.nlls = []


    def model_with_skip(self, layers_to_skip):
        # Modify the model to skip a specific layer
        import copy
        if isinstance(layers_to_skip, int):
            layers_to_skip = [layers_to_skip]
        modified_model = copy.deepcopy(self.model)
        modified_model.transformer.h = nn.ModuleList(
            [layer for i, layer in enumerate(modified_model.transformer.h) if i not in layers_to_skip]
        )
        self.model = modified_model

    def model_with_swap(self,):
        import copy
        modified_model = copy.deepcopy(self.model)
        new_order = list(range(len(modified_model.transformer.h)))
        new_order[self.swap_layer], new_order[self.swap_layer + 1] = new_order[self.swap_layer + 1], new_order[self.swap_layer]
        modified_model.transformer.h = nn.ModuleList(
            [self.model.transformer.h[i] for i in new_order]
        )
        self.model = modified_model

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        target_len = batch['target_len']

        # Get model outputs without computing internal loss
        outputs = self.model(input_ids=input_ids)
        # Extract logits
        logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Flatten logits and labels
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # Shape: [total_tokens, vocab_size]
        shift_labels = shift_labels.view(-1)  # Shape: [total_tokens]

        # Compute per-token loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_per_token = loss_fct(shift_logits, shift_labels)  # Shape: [total_tokens]

        # Store per-token losses (1D tensor)
        self.per_token_losses = torch.cat([self.per_token_losses, loss_per_token[-target_len:].detach().cpu()], dim=0)

        # Compute total loss
        total_loss = loss_per_token.sum() / (shift_labels != -100).sum()

        # Assertion to verify the total loss matches model's loss
        with torch.no_grad():
            outputs_with_loss = self.model(input_ids=input_ids, labels=target_ids)
            model_loss = outputs_with_loss.loss
            assert torch.allclose(total_loss, model_loss, atol=1e-4), \
                f"Total loss {total_loss.item()} does not match model loss {model_loss.item()}"

        self.nlls.append(total_loss)
        return total_loss

    def on_test_epoch_end(self):
        # Compute per-token perplexities
        per_token_perplexities = torch.exp(self.per_token_losses)  # Shape: [total_tokens]

        # Compute overall test perplexity
        total_loss = self.per_token_losses.mean()
        test_perplexity = torch.exp(total_loss)

        # Log overall test perplexity
        self.log('test_perplexity', test_perplexity)

        # Save per-token perplexities to a file
        per_token_perplexities_file = f'per_token_perplexities_{self.current_epoch}.pt'
        torch.save(per_token_perplexities, per_token_perplexities_file)

        # Create an artifact
        artifact = wandb.Artifact(
            name=f'per_token_perplexities_{self.current_epoch}',
            type='per_token_data'
        )
        artifact.add_file(per_token_perplexities_file)
        self.logger.experiment.log_artifact(artifact)
        self.logger.experiment.log({
            'per_token_perplexity_histogram': wandb.Histogram(per_token_perplexities.cpu().numpy())
        })
        # ppl = torch.exp(torch.stack(self.nlls).mean())
        # self.log('test_perplexity', ppl)
        # print(f'Test Perplexity: {ppl.item():.2f}')

    def configure_optimizers(self):
        # Not needed since we're only evaluating
        return None
