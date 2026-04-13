from datasets import load_dataset
from core.config import Config
from torch.utils.data import Dataset
from itertools import chain
import torch
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, config: Config, split="train"):

        super().__init__()
        self.config = config
        self.block = config.data.block_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = load_dataset(config.data.dataset, split=split)

        def tokenize_fn(example):
            return self.tokenizer(example["text"])

        tokenized_dataset = dataset.map(
            tokenize_fn=tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset...",
        )

        all_tokens = list(chain.from_iterable(tokenized_dataset["input_ids"]))

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens)) // self.config.data.block_size

    def __getitem__(self, idx):
        start = idx * self.config.data.block_size
        end = start + self.config.data.block_size
        chunk = self.tokens[start:end]
        return chunk
