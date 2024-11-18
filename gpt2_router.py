import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pytorch_lightning as pl
import copy


class Router(nn.Module):
    def __init__(self, input_dim, num_layers, iters, k):
        super(Router, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim + iters)  # Layer normalization
        self.fc = nn.Linear(input_dim + iters, num_layers, bias=False)

        # Custom weight initialization to introduce a bias toward the correct order
        with torch.no_grad():
            # Initialize with small values
            self.fc.weight *= 0.01

            # Add an identity-like initialization based on the number of layers and iterations
            min_size = min(iters, num_layers)
            self.fc.weight[:min_size, input_dim:input_dim + min_size] = torch.eye(min_size)

    def forward(self, x, topk=None):
        x = self.layer_norm(x)  # Normalize the inputs before feeding into the fully connected layer
        logits = self.fc(x)
        if topk is not None:
            topk_values, topk_indices = torch.topk(logits, topk, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            return probs, topk_indices
        else:
            probs = torch.softmax(logits, dim=-1)
            return probs


# Step 2: Define the RouterBlock to replace individual transformer blocks
class RouterBlock(nn.Module):
    def __init__(self, original_layers, input_dim, iters):
        super(RouterBlock, self).__init__()
        self.num_layers = len(original_layers)
        self.iters = iters
        self.router = Router(input_dim, self.num_layers, iters, k=1)  # No top-k, soft routing for all layers
        self.original_layers = original_layers

        for layer in self.original_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, hidden_states, attention_mask, iteration, inference_mode=False):
        # Mask padding tokens in attention_mask (if not None)
        if attention_mask is not None:
            mask = attention_mask.view(hidden_states.size(0), hidden_states.size(1), 1).expand_as(hidden_states)
            hidden_states = hidden_states * mask

        batch_size, seq_len, hidden_dim = hidden_states.size()

        # Create a one-hot encoding of the iteration and concatenate it with the router inputs
        iteration_one_hot = F.one_hot(torch.tensor(iteration), num_classes=self.iters).float().to(
            hidden_states.device)
        iteration_one_hot = iteration_one_hot.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
        # Run the router for all tokens in the sequence
        router_inputs = torch.cat((hidden_states, iteration_one_hot),
                                  dim=-1)  # Concatenate hidden states with iteration one-hot

        if inference_mode:
            probs, topk_experts = self.router(router_inputs, topk=1)  # TODO: change to self.router.k
            # Find the most common element
            # unique_elements, counts = torch.unique(topk_experts, return_counts=True)
            # layer_idx = unique_elements[torch.argmax(counts)].item()
            layer_idx = topk_experts[0, -1, 0]
            # Process each layer in the block for all tokens
            updated_hidden_states = torch.zeros_like(hidden_states)
            layer_output, _ = self.original_layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask.float() if attention_mask is not None else None,
                use_cache=False
            )
            # Multiply layer output by the corresponding probabilities
            updated_hidden_states += layer_output

        else:
            probs = self.router(router_inputs)

            # Process each layer in the block for all tokens
            updated_hidden_states = torch.zeros_like(hidden_states)
            for layer_idx in range(self.num_layers):
                layer_output, _ = self.original_layers[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask.float() if attention_mask is not None else None,
                    use_cache=False
                )
                # Multiply layer output by the corresponding probabilities
                updated_hidden_states += probs[:, :, layer_idx].unsqueeze(-1) * layer_output

        probs = probs.detach().cpu().numpy()
        return updated_hidden_states, probs


# Step 3: Add the Router to GPT2 with Blocked Layers
class GPT2WithRouter(pl.LightningModule):
    def __init__(self, model_name='gpt2', k=2, block_size=6, iters=6, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.k = k
        self.iters = iters
        self.learning_rate = learning_rate

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Pretrained Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.eos_token_id
        )

        for param in self.model.parameters():
            param.requires_grad = False

        # Add the router to blocks of transformer layers
        num_layers = len(self.model.transformer.h)
        input_dim = self.model.transformer.h[0].mlp.c_fc.nx
        self.block_routers = nn.ModuleList()

        # Split layers into blocks of size `block_size`
        for i in range(0, num_layers, block_size):
            block_layers = self.model.transformer.h[i:i + block_size]
            print(f"Block {i}: layers {list(range(len(self.model.transformer.h)))[i: i + block_size]}")
            router_block = RouterBlock(block_layers, input_dim, self.iters)
            self.block_routers.append(router_block)

        self.nlls = []

    def forward(self, input_ids, attention_mask=None, inference_mode=False):
        assert attention_mask is None
        # Embed tokens and add position embeddings
        hidden_states = self.model.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds

        # Apply dropout after embeddings
        hidden_states = self.model.transformer.drop(hidden_states)

        all_router_probs_blocks = []
        num_blocks = len(self.block_routers)

        # Pass through blocks
        for block_idx in range(num_blocks):
            all_router_probs_blocks.append([])
            router_block = self.block_routers[block_idx]
            for i in range(self.iters):
                hidden_states_new, router_probs = router_block(hidden_states, attention_mask, i,
                                                              inference_mode=inference_mode)
                hidden_states = hidden_states_new
                all_router_probs_blocks[block_idx].append(router_probs)

        # Apply final layer norm
        hidden_states = self.model.transformer.ln_f(hidden_states)

        # Compute logits for next token prediction
        logits = self.model.lm_head(hidden_states)

        return logits, all_router_probs_blocks

    def _shared_step(self, batch, batch_idx, inference_mode=False):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']

        outputs, all_router_probs = self(input_ids=input_ids, inference_mode=inference_mode)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Log router probabilities in a more manageable way
        for block_idx, block_prob in enumerate(all_router_probs):
            for i, prob in enumerate(block_prob):
                prob = prob.mean(axis=(0, 1))  # average over batch and sequence
                for layer_idx, p in enumerate(prob):
                    self.log(f"avg_router_probs_b_{block_idx}_i_{i}_l_{layer_idx}", p, prog_bar=True, logger=True)

        # Compute the loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, inference_mode=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.nlls.append(loss)
        return loss

    def on_test_epoch_end(self):
        ppl = torch.exp(torch.stack(self.nlls).mean())
        self.log('test_perplexity', ppl)
        print(f'Test Perplexity: {ppl.item():.2f}')

    def configure_optimizers(self):
        # Only train the router, not the transformer layers
        optimizer = torch.optim.AdamW(self.block_routers.parameters(), lr=self.learning_rate)
        return optimizer

# Step 4: Test the modified model
if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the original GPT2 model
    original_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize the model with router and block size set to 1
    routed_model = GPT2WithRouter(model_name=model_name, k=1, block_size=1)

    # Tokenize some input text
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt")

    # Run a forward pass with the original model
    with torch.no_grad():
        original_logits = original_model(inputs['input_ids']).logits

    # Run a forward pass with the routed model
    with torch.no_grad():
        routed_logits, _ = routed_model(inputs['input_ids'])

    # Check if the logits from both models are identical
    if torch.allclose(original_logits, routed_logits, atol=1e-5):
        print("The logits from the original model and the routed model are identical.")
    else:
        print("The logits from the original model and the routed model are NOT identical.")

    # Print a few logits to compare visually
    print("Original Logits:", original_logits[0, :5, :5])
    print("Routed Logits:", routed_logits[0, :5, :5])
