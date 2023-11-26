"""
    Collate function for batch processing
"""

import torch
from transformers import DataCollatorWithPadding

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorWithPadding):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])
        tk_ids = torch.tensor(tk_ids)
        attn_masks = torch.tensor(attn_masks)
        lbs = torch.tensor(lbs)
        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
