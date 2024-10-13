# data_module.py

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, encodings, block_size, stride):
        self.seq_len = encodings.input_ids.size(1)
        self.input_ids = encodings.input_ids
        self.block_size = block_size
        self.stride = stride

    def __len__(self):
        return self.seq_len//self.stride + 1

    def __getitem__(self, idx):
        begin_loc = idx * self.stride
        end_loc = min(begin_loc + self.block_size, self.seq_len)
        input_ids = self.input_ids[:, begin_loc:end_loc]
        target_ids = self.input_ids.clone()
        target_len = end_loc - ((idx - 1) * self.stride + self.block_size) #  this equals to stride except on the last iter
        target_ids[:, :-target_len] = -100  # don't use the overlapping tokens from last iter as target because they were already used
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }

class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name, block_size=1024, stride=512):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.stride = stride

    def prepare_data(self):
        load_dataset('wikitext', 'wikitext-2-raw-v1')

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        # Concatenate all test texts
        test_text = "\n\n".join(test_dataset['text'])

        # Tokenize the concatenated text
        encodings = tokenizer(test_text, return_tensors='pt')

        # Create the custom dataset
        self.test_dataset = TextDataset(
            encodings, 
            block_size=self.block_size, 
            stride=self.stride
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=96)
