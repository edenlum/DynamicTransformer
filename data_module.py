# data_module.py

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size, stride):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.inputs = []
        self.targets = []

        # Tokenize the entire text and create sliding windows with block_size and stride
        encodings = tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0)
        for i in range(0, len(encodings) - block_size + 1, stride):
            input_ids = encodings[i:i + block_size]
            target_ids = input_ids.clone()
            target_ids[:-self.stride] = -100
            if len(input_ids) == block_size:
                self.inputs.append(input_ids)
                self.targets.append(target_ids)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return  {
            'input_ids': self.inputs[idx],
            'target_ids': self.targets[idx],
            'target_len': self.stride
        }

class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name, block_size=1024, stride=512, split='test', batch_size=8):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.stride = stride
        self.split = split
        self.batch_size = batch_size

    def prepare_data(self):
        load_dataset('wikitext', 'wikitext-2-raw-v1')

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=self.split)
        text = "\n\n".join(dataset['text'])
        self.dataset = TextDataset(text, tokenizer, self.block_size, self.stride)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=8)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8)

if __name__ == "__main__":
    # Test the WikiTextDataModule
    data_module = WikiTextDataModule(tokenizer_name='gpt2', block_size=1024, stride=512, split='test', batch_size=2)
    data_module.prepare_data()
    data_module.setup()
    test_loader = data_module.train_dataloader()

    # Print the shapes of the data
    for batch in test_loader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Target IDs shape: {batch['target_ids'].shape}")
        print(f"Target Length: {batch['target_len']}")
        break
