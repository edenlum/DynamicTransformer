from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import copy
import torch.nn as nn

model_name = "gpt2"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

dataset = load_dataset("glue", "sst2", split="validation")
texts = dataset["sentence"]
labels = dataset["label"]

def encode_texts(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    return inputs.input_ids, inputs.attention_mask

def prepare_dataloader(texts, labels, tokenizer, batch_size=32):
    input_ids, attention_mask = encode_texts(texts, tokenizer)
    dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size)

def evaluate_sample(model, input_id, attention_mask, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_id.unsqueeze(0).to(device), attention_mask=attention_mask.unsqueeze(0).to(device))
    return torch.argmax(output.logits, dim=-1).item()

def remove_layer(model, layers_to_remove):
    """Removes the specified layers from the model."""
    if not isinstance(layers_to_remove, list):
        layers_to_remove = [layers_to_remove]
    modified_model = copy.deepcopy(model)
    modified_model.distilbert.transformer.layer = nn.ModuleList(
        [layer for i, layer in enumerate(modified_model.distilbert.transformer.layer) if i not in layers_to_remove]
    )
    return modified_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = prepare_dataloader(texts, labels, tokenizer)

def test_hypothesis(model, dataloader, device, layer_to_remove):
    stable_count = 0
    total_count = 0
    correct_count = 0
    correct_count_modified = 0
    unstable_indices = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        for i in range(input_ids.size(0)):
            original_output = evaluate_sample(model, input_ids[i], attention_mask[i], device)
            if original_output == labels[i].item():
                correct_count += 1

            modified_model = remove_layer(model, layer_to_remove)
            modified_output = evaluate_sample(modified_model, input_ids[i], attention_mask[i], device)
            if modified_output == labels[i].item():
                correct_count_modified += 1

            if original_output == modified_output:
                stable_count += 1
            else:
                unstable_indices.append(batch_idx * dataloader.batch_size + i)

            total_count += 1

    print(f"Stable samples: {stable_count}")
    print(f"Total samples: {total_count}")
    print(f"Stability rate: {stable_count / total_count:.4f}")
    print(f"Correct samples: {correct_count}")
    print(f"Correct samples modified: {correct_count_modified}")

    return unstable_indices

layer_wrong_idx_dict = {}
for layer_to_remove in range(len(model.distilbert.transformer.layer)):
    print(f"Removing layer {layer_to_remove}")
    layer_wrong_idx_dict[layer_to_remove] = test_hypothesis(model, dataloader, device, [layer_to_remove])
    print(layer_wrong_idx_dict[layer_to_remove])