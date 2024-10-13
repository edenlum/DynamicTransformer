# data_module.py

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, encodings, block_size):
        self.input_ids = encodings.input_ids[0]
        self.block_size = block_size

    def __len__(self):
        return (len(self.input_ids) - 1) // self.block_size + 1

    def __getitem__(self, idx):
        begin_loc = idx * self.block_size
        end_loc = min(begin_loc + self.block_size, len(self.input_ids))
        input_ids = self.input_ids[begin_loc:end_loc]
        return {
            'input_ids': input_ids,
            'begin_loc': begin_loc,
            'end_loc': end_loc,
            'trg_len': end_loc - begin_loc,
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
        self.test_dataset = TextDataset(encodings, block_size=self.stride)

        # Store the total number of tokens
        self.test_dataset_size = len(self.test_dataset.input_ids)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=96)
