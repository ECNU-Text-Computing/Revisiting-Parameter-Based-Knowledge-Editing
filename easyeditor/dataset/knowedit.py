import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class KnowEditDataset(Dataset):
    """
    Dataset of factual knowledge based on KnowEdit.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 40

        # For Meta Training
        if config is not None and hasattr(config, 'tokenizer_name'):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
            # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            if 'qwen' in config.model_name.lower():
                tokenizer.eos_token='<|endoftext|>'
                tokenizer.pad_token='<|endoftext|>'
                tokenizer.unk_token='<|endoftext|>'
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

            """ # from zsre
            tokenizer = AutoTokenizer.from_pretrained(
                tok_name, trust_remote_code=True
            )
            if 'gpt' in tok_name.lower(): # isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif 'llama' in tok_name.lower():
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            elif 'qwen' in tok_name.lower():
                tokenizer.eos_token='<|endoftext|>'
                tokenizer.pad_token='<|endoftext|>'
                tokenizer.unk_token='<|endoftext|>'
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            elif 'mistral' in tok_name.lower():
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('MistralTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer
            """

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            data.append(
                {
                    "subject":record["subject"] if "subject" in record else record["concept"],
                    "prompt": record["prompt"] if "prompt" in record else (record["text"] if "text" in record else record["src"]),
                    "target_new": record["target_new"] if "target_new" in record else (record["labels"] if "labels" in record else record["answers"][0]),
                    "rephrase_prompt": record["rephrase"] if "rephrase" in record else record["rephrase_prompt"] if "rephrase_prompt" in record else None,
                    "ground_truth": record["ground_truth"] if "ground_truth" in record else None,
                    "portability_r": record["portability"]["Reasoning"] if "portability" in record and "Reasoning" in record["portability"] else None,
                    "portability_s": record["portability"]["Subject_Aliasing"] if "portability" in record and "Subject_Aliasing" in record["portability"] else None,
                    "portability_l":record["portability"]["Logical_Generalization"] if "portability" in record and "Logical_Generalization" in record["portability"] else None,
                    "locality_rs": record["locality"]["Relation_Specificity"] if "locality" in record and "Relation_Specificity" in record["locality"]
                    else [{"prompt": record["loc"].replace("nq question: ", ""), "ground_truth": [record["loc_ans"]]}] if "loc" in record and "loc_ans" in record
                    else None,
                    "locality_f": record["locality"]["Forgetfulness"] if "locality" in record and "Forgetfulness" in record["locality"] else None
                }
            )
        if 'start_index' in kwargs and 'end_index' in kwargs:
            start_index=kwargs["start_index"]
            end_index=kwargs["end_index"]
            if start_index is not None and end_index is not None:
                data = data[start_index:end_index]
                print("这是data")
        if size is not None:
            data = data[:size]
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        loc_data = [b["locality_rs"] if b["locality_rs"]!=None else b["locality_f"] for b in batch]
        loc=[l[0]["prompt"] if isinstance(l[0]["prompt"],str) else l[0]["prompt"][0] for l in loc_data]
        loc_ans = [l[0]["ground_truth"][0] if isinstance(l[0]["ground_truth"][0],str) else l[0]["ground_truth"][0][0] for l in loc_data]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        loc_data = [b["locality_rs"] if b["locality_rs"]!=None else b["locality_f"] for b in batch]
        loc=[l[0]["prompt"] if isinstance(l[0]["prompt"],str) else l[0]["prompt"][0] for l in loc_data]

        loc_ans = [l[0]["ground_truth"] if isinstance(l[0]["ground_truth"][0],str) else l[0]["ground_truth"][0] for l in loc_data]
        loc_ans = [l if isinstance(l,str) else l[0] for l in loc_ans]

        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels


        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO
        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
    
