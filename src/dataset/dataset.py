"""
    Dataset loading and processing
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import AutoTokenizer

from src.args import Config
from src.utils.data import span_to_label, span_list_to_dict
from .batch import pack_instances

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

MASKED_LB_ID = -100
MAX_TOKENS = 510

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text: Optional[List[List[str]]] = None, lbs: Optional[List[List[str]]] = None):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._token_ids = None
        self._attn_masks = None
        self._bert_lbs = None

        self._partition = None

        self.data_instances = None

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def prepare(self, config: Config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """
        assert partition in ["train", "valid", "test"], ValueError(
            f"Argument `partition` should be one of 'train', 'valid' or 'test'!"
        )
        self._partition = partition
        if partition == "test":
            file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.csv"))
        else:
            file_path = os.path.normpath(os.path.join(config.data_dir, config.experiment, f"{partition}.csv"))
        logger.info(f"Loading data file: {file_path}")
        self._text, self._lbs = load_data_from_csv(file_path)

        logger.info("Encoding sequences...")
        self.encode(config.bert_model_name_or_path, {lb: idx for idx, lb in enumerate(config.bio_label_types)})

        logger.info(f"Data loaded.")

        self.data_instances = pack_instances(
            bert_tk_ids=self._token_ids,
            bert_attn_masks=self._attn_masks,
            bert_lbs=self._bert_lbs,
        )
        return self

    def encode(self, tokenizer_name: str, lb2idx: dict):
        """
        Build BERT token as model input

        Parameters
        ----------
        tokenizer_name: str
            the name of the assigned Huggingface tokenizer
        lb2idx: dict
            a dictionary that maps the str labels to indices

        Returns
        -------
        self
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        tokenized_text = tokenizer(self._text, 
                                   add_special_tokens=True, 
                                   padding="max_length",
                                   max_length=MAX_TOKENS, 
                                   truncation=True)

        self._token_ids = tokenized_text.input_ids
        self._attn_masks = tokenized_text.attention_mask
        self._bert_lbs = [lb2idx[lb] for lb in self._lbs]

        return self


def load_data_from_csv(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: str
        file directory

    """
    
    df = pd.read_csv(file_dir, encoding="utf-8", header=0)

    tk_seqs = list()
    lbs_list = list()

    for _, inst in df.iterrows():
        # get tokens
        tk_seqs.append(inst["text"])

        # get true labels
        lbs_list.append(inst["label"])

    return tk_seqs, lbs_list
