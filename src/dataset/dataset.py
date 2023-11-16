"""
    Dataset loading and processing
"""

import os
import json
import logging
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer

from src.args import Config
from src.utils.data import span_to_label, span_list_to_dict
from .batch import pack_instances

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

MASKED_LB_ID = -100


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

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))
        logger.info(f"Loading data file: {file_path}")
        self._text, self._lbs = load_data_from_json(file_path)

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
        Build BERT token masks as model input

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
        tokenized_text = tokenizer(self._text, add_special_tokens=True, is_split_into_words=True)

        self._token_ids = tokenized_text.input_ids
        self._attn_masks = tokenized_text.attention_mask

        bert_lbs_list = list()

        # Update label sequence to match the BERT tokenization
        for i, token_ids in enumerate(self._token_ids):
            bert_lbs = []
            for j, token_id in enumerate(token_ids):
                word_id = tokenized_text.word_ids(i)[j]
                if token_id == tokenizer.cls_token_id or token_id == tokenizer.sep_token_id:
                    bert_lbs.append(MASKED_LB_ID)
                elif j > 0 and word_id == tokenized_text.word_ids(i)[j-1]:
                    bert_lbs.append(MASKED_LB_ID)
                else:
                    label = self._lbs[i][word_id]
                    bert_lbs.append(lb2idx[label] if label in lb2idx else lb2idx["O"])
            bert_lbs_list.append(bert_lbs)

        for tks, lbs in zip(self._token_ids, bert_lbs_list):
            assert len(tks) == len(lbs), ValueError(
                f"Length of token ids ({len(tks)}) and labels ({len(lbs)}) mismatch!"
            )

        self._bert_lbs = bert_lbs_list

        return self


def load_data_from_json(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: str
        file directory

    """
    with open(file_dir, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    tk_seqs = list()
    lbs_list = list()

    for inst in data_list:
        # get tokens
        tk_seqs.append(inst["text"])

        # get true labels
        lbs = span_to_label(span_list_to_dict(inst["label"]), inst["text"])
        lbs_list.append(lbs)

    return tk_seqs, lbs_list
