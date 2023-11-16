"""
    Collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch
        padded_ids = self.tokenizer.pad({"input_ids": tk_ids}, return_attention_mask=True, padding="longest", return_tensors="pt")
        tk_ids = padded_ids["input_ids"]
        attn_masks = padded_ids["attention_mask"]
        lbs = torch.nn.utils.rnn.pad_sequence([torch.tensor(lb) for lb in lbs], batch_first=True, padding_value=self.label_pad_token_id)

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
