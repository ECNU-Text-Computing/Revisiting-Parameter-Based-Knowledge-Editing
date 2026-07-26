import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
import os
import os.path
import sys
import json
import random
from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np


import argparse
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer


class ELKENDataset(Dataset):
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

        with open(zsre_loc, "r") as f:
            raw = json.load(f)  # 是个 list

        data = []  # 最终是一个列表，每个元素是一个字典
        for record in raw:
            subjects = record["triples"][0]["subject"] 
            prompts = record["triples"][0]["prompt"]
            #subjects = [item["subject"] for item in record["triples"]]
            #prompts = [item["prompt"] for item in record["triples"]]
            #for subject,prompt in zip(subjects,prompts):
            #    assert subject in prompt, print(f'Subject:{subject} do not exist in prompt: {prompt}')
            #prompts=[]
            # for item in record["triples"]:
            #     if "prompt" not in item:
            #         print("Missing 'prompt' in item:", item)
            #         print("Full record:", record)
            #         raise KeyError("'prompt' not found in item")
            #     prompts.append(item["prompt"])
            target_news = record["triples"][0]["target_new"]
            fact_question = [item["question"] for item in record["fact"]["qas"]]
            fact_answer =  [item["answer"]["name"] for item in record["fact"]["qas"]]

            local_fact = [
                    item["question"] 
                    for item in record.get("fact", {}).get("local_qas", [])
                ]
            local_fact_answer = [
                    item["answer"]["name"]
                    for item in record.get("fact", {}).get("local_qas", [])
                    if "answer" in item and "name" in item["answer"]
                ]


            data.append({
                "subjects": subjects,
                "prompts": prompts,
                "target_new": target_news,
                "fact_question": fact_question,
                "fact_answer":fact_answer,
                "local_fact_question":local_fact,
                "local_fact_answer":local_fact_answer
            })

        if size is not None:
            data = data[:size]
        
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    
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
                tokenizer.eos_token = '<|endoftext|>'
                tokenizer.pad_token = '<|endoftext|>'
                tokenizer.unk_token = '<|endoftext|>'
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        with open(zsre_loc, "r") as f:
            # raw = json.load(f)
            raw = [json.loads(line.strip()) for line in f if line.strip()]

        data = []
        for i, record in enumerate(raw):
            data.append(
                {
                    "subject": record["subject"] if "subject" in record else record["concept"],
                    "prompt": record["prompt"] if "prompt" in record else record["text"],
                    "rephrase_prompt": record["rephrase"] if "rephrase" in record else record[
                        "rephrase_prompt"] if "rephrase_prompt" in record else record["prompt"],
                    "target_new": record["target_new"] if "target_new" in record else record["labels"],
                    "ground_truth": record["ground_truth"] if "ground_truth" in record else None,
                    "portability_r": record["portability"]["Reasoning"] if "portability" in record and "Reasoning" in
                                                                           record["portability"] else None,
                    "portability_s": record["portability"][
                        "Subject_Aliasing"] if "portability" in record and "Subject_Aliasing" in record[
                        "portability"] else None,
                    "portability_l": record["portability"][
                        "Logical_Generalization"] if "portability" in record and "Logical_Generalization" in record[
                        "portability"] else None,
                    "locality_rs": record["locality"]["Relation_Specificity"] if "Relation_Specificity" in record[
                        "locality"] else None,
                    "locality_f": record["locality"]["Forgetfulness"] if "Forgetfulness" in record["locality"] else None
                }
            )

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
        loc_data = [b["locality_rs"] if b["locality_rs"] != None else b["locality_f"] for b in batch]
        loc = [l[0]["prompt"] if isinstance(l[0]["prompt"], str) else l[0]["prompt"][0] for l in loc_data]
        loc_ans = [l[0]["ground_truth"][0] if isinstance(l[0]["ground_truth"][0], str) else l[0]["ground_truth"][0][0]
                   for l in loc_data]

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
        loc_data = [b["locality_rs"] if b["locality_rs"] != None else b["locality_f"] for b in batch]
        loc = [l[0]["prompt"] if isinstance(l[0]["prompt"], str) else l[0]["prompt"][0] for l in loc_data]

        loc_ans = [l[0]["ground_truth"] if isinstance(l[0]["ground_truth"][0], str) else l[0]["ground_truth"][0] for l
                   in loc_data]
        loc_ans = [l if isinstance(l, str) else l[0] for l in loc_ans]

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


# def load_data(data_dir,datatype,evaluation):
def load_data(data_dir,datatype):
    

    if datatype == 'counterfact' or datatype == 'recent' or datatype == 'zsre':
        datas = KnowEditDataset(data_dir)
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        
        rephrase_prompts = [data['rephrase_prompt'] for data in datas]

        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]

        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]
        
        portability_data = [portability_r,portability_s,portability_l]
        portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            }
        }
        return {'prompts': prompts, 'subjects': subjects, 'target_new': target_new,'rephrase_prompts':rephrase_prompts, 'locality_inputs':locality_inputs,'portability_inputs':portability_inputs}


    if datatype == 'ELKEN':
        datas = ELKENDataset(data_dir)
        print(datas)
        all_case_ids = []
        all_prompts=[]
        all_target_new = []
        all_subjects = []
        all_locality_prompts = []
        all_locality_ans = []
        all_portability_prompts = []
        all_portability_ans = []
        prompt_len=[]
        for index,event in enumerate(datas):  # 每一个event有多个prompt进行顺序编辑
            case_id = str(index)
            all_case_ids.append(case_id)
            prompts = event["prompts"][0]
            prompt_len.append(len(prompts))
            all_prompts.append(event["prompts"][0])
            all_target_new.append(event["target_new"][0])
            all_subjects.append(event["subjects"][0])

            local_fact_question = event["local_fact_question"] # 再改为一个列表套列表
            local_fact_answer = event["local_fact_answer"] # 再改为一个列表套列表

            fact_question = event["fact_question"]  # 再改为一个列表套列表
            fact_answer = event["fact_answer"] # 再改为一个列表套列表


            locality_prompts = [None]*len(prompts)
            locality_prompts[-1] = local_fact_question if len(local_fact_question)>0 else None
            all_locality_prompts.extend(locality_prompts)

            locality_ans = [None]*len(prompts)
            locality_ans[-1] = local_fact_answer if len(local_fact_answer)>0 else None 
            all_locality_ans.extend(locality_ans)

            portability_prompts = [None]*len(prompts)
            portability_prompts[-1] = fact_question if len(fact_question)>0 else None
            all_portability_prompts.extend(portability_prompts)

            portability_ans = [None]*len(prompts)
            portability_ans[-1] = fact_answer if len(fact_answer)>0 else None
            all_portability_ans.extend(portability_ans)

        assert len(all_prompts) == len(all_locality_prompts) == len(all_locality_ans) 
        
        locality_inputs = {
            'Locality_Fact':{
                'prompt': all_locality_prompts,
                'ground_truth': all_locality_ans
            }
        }

        assert len(all_prompts) == len(all_portability_prompts) == len(all_portability_ans) 
        
        portability_inputs = {
            'Portability_Fact':{
                'prompt': all_portability_prompts,
                'ground_truth': all_portability_ans
            }
        }
    return {'case_ids':all_case_ids,'prompts': all_prompts, 'subjects': all_subjects, 'target_new': all_target_new,'locality_inputs':locality_inputs,'portability_inputs':portability_inputs}





def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'portability': {},
        'locality': {}
    }
    for prompt, target_new_ in zip(prompts,  target_new)
    ]

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        else:
            assert len(kwargs['loc_prompts']) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts,]

        for i, request in enumerate(requests):
            request.update(
                {
                    'rephrase_prompt': rephrase_prompts[i],
                }
            )
    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            #assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            #== len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            #assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            #== len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
    return requests

