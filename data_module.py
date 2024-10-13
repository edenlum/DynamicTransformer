import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import DataCollatorWithPadding



class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=1, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def prepare_data(self):
        # Download data if needed
        load_dataset('wikitext', 'wikitext-2-raw-v1')

    def setup(self, stage=None):
        # Load and tokenize dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )

        self.test_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
        )

        # Set dataset format to PyTorch tensors
        self.test_dataset.set_format(type='torch')

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer, padding='longest'),
            num_workers=96
        )
