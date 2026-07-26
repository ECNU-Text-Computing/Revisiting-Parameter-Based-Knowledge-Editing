"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
# LORA wrapper check: duck-type instead of hard import from melo
def _is_lora_wrapped(model):
    """Check if model is wrapped in a LoRA/QLoRA adapter."""
    return hasattr(model, 'model') and hasattr(model, 'peft_config')

import typing
from itertools import chain
from typing import List, Optional

import os
import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_prediction_acc_LLM_judge,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    OOD_PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if _is_lora_wrapped(model):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, contain_score, gen_content = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_acc": acc,
            f"{key}_contain_acc": contain_score,
            f"{key}_gen_content": gen_content
        }
    else:  # traditional evaluation 
        if eval_metric == 'ppl':
            ppl = PPL(model, tok, prompt, target_new, device)
            ret = {
                f"{key}_ppl": ppl
            }
        elif eval_metric == 'ood_ppl':
            ans = OOD_PPL(model, tok, prompt, target_new, device)
            ret = {
                f"ood_acc": ans
            }
        elif hparams.alg_name=="GRACE":
            # ppl = PPL(model, tok, prompt, target_new, device)
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
            f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
            ret = {
                f"{key}_acc": acc,
                # f"{key}_PPL": ppl,
                f"{key}_F1":f1     
            }        
        else:  # teacher-forcing evaluation
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
            ret = {
                f"{key}_acc": acc
            }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, contain_score, gen_content = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        ret = {
            f"{locality_key}_acc": acc,
            f"{locality_key}_contain_acc": contain_score,
            f"{locality_key}_gen_content": gen_content,          
        }
    else:  # traditional evaluation 
        if 't5' in model_name.lower():
            loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        else:
            loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')
        if type(loc_tokens) is not list:
            loc_tokens = [loc_tokens,]

        ret = {
            f"{locality_key}_output": loc_tokens
        }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    # using real-world evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "LLM-judge":
        acc, contain_score, gen_content  = test_prediction_acc_LLM_judge(model, tok, hparams, prompt, ground_truth, device, locality=False)
        ret = {
            f"{portability_key}_acc": acc,
            f"{portability_key}_contain_acc": contain_score,
            f"{portability_key}_gen_content": gen_content
        }
    else:  # traditional evaluation
        if 't5' in model_name.lower():
            portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
        else:
            portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

        ret = {
            f"{portability_key}_acc": portability_correct
        }
    return ret




import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
import json
class RetrieveTool:
    def __init__(self, retriever_path, memory_path, device):
        # 仅使用 retriever_path 初始化 SentenceTransformer 模型
        self.retriever = AutoModel.from_pretrained(retriever_path).to(device)
        self.retriever_tok = AutoTokenizer.from_pretrained(retriever_path)
        self.device = device
        # 读取 memory 数据
        
        memory = []
        with open(memory_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            for entry in data:
                question = entry["question"]
                answer = entry["answer"]
                memory.append(question+" "+answer)
        
        self.memory = memory
        self.get_sent_embeddings(self.memory)

    def mean_pooling(self, token_embeddings, mask):
        # 使用 mask 计算平均池化
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_sent_embeddings(self, memory, BSZ=1):
        # 生成 memory 的嵌入
        all_embs = []
        for i in tqdm(range(0, len(memory), BSZ)):
            sent_batch = memory[i:i + BSZ]
            #start_time = time.time()   
            inputs = self.retriever_tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.retriever(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            all_embs.append(embeddings.cpu())
            
    
        all_embs = torch.vstack(all_embs)
        self.memory_embedding = all_embs

    def retrieve(self, query,  k=1):

        inputs = self.retriever_tok([query], padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.retriever(**inputs)
            query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()


        sim = (query_emb @ self.memory_embedding.T)[0]
        knn = sim.topk(k, largest=True)
        fact_ids = knn.indices

        # Retrieve multiple facts
        retrieved_facts = [self.memory[fact_id] for fact_id in fact_ids]
        similarity_scores = knn.values.detach().cpu().numpy().tolist()

        return retrieved_facts, similarity_scores  # return a list


global global_retriever
# Set EASYEDIT_RETRIEVER_PATH and EASYEDIT_RETRIEVER_DATA environment variables,
# or configure paths below. The retriever is used for ICL-based evaluation.
_retriever_path = os.environ.get(
    'EASYEDIT_RETRIEVER_PATH',
    '/path/to/contriever-msmarco'
)
_retriever_data = os.environ.get(
    'EASYEDIT_RETRIEVER_DATA',
    '/path/to/retrieval-corpus.json'
)
global_retriever = RetrieveTool(_retriever_path, _retriever_data, "cuda:0")


def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.

    # sx: 修改这里，target_prompt，target_new是检索得到的 -》 new_fact
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    retrieved_facts, similarity_scores = global_retriever.retrieve(prompt, k=1)
    target_prompt = retrieved_facts[0]
    new_fact = f'New Fact: {retrieved_facts[0]}\nPrompt: {prompt}'

    
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    #new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        acc, contain_score, gen_content = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
        ret = {
            f"rewrite_acc": acc,
            f"rewrite_contain_acc": contain_score,
            f"rewrite_gen_content": gen_content
        }
        

    if rephrase is not None:
        acc, contain_score, gen_content = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret["rephrase_acc"]=acc
        ret["rephrase_contain_acc"]=contain_score
        ret["rephrase_gen_content"]=gen_content
    ret['locality'] = {}
    ret['portability'] = {}
    """
    def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target, # 是个列表
        x,
        neighborhood=False  
    )-> typing.Dict:
    """

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            x = [f"New Fact: {prompt} {target_new}\nPrompt: {cur_question}" for cur_question in record['locality'][locality_key]['prompt']]
            acc, contain_score, gen_content = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality'][locality_key]['ground_truth'],
                                                    x,
                                                    neighborhood=True)

            key_ret = {
                f"{locality_key}_acc": acc,
                f"{locality_key}_contain_acc": contain_score,
                f"{locality_key}_gen_content": gen_content,          
            }
            ret['locality'].update(key_ret)

      
    if 'portability' in record.keys() and any(record['portability']):
        
        for portability_key in record['portability'].keys():
            icl_input = icl_examples
            x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            x = [f"{x_prefix}{x_p}" for x_p in record['portability'][portability_key]['prompt']]
            acc, contain_score, gen_content = icl_lm_eval(model, model_name, hparams, tok, icl_input, record['portability'][portability_key]['ground_truth'],
                                                      x)
            key_ret = {
                f"{portability_key}_acc": acc,
                f"{portability_key}_contain_acc": contain_score,
                f"{portability_key}_gen_content": gen_content,          
            }
            ret['portability'].update(key_ret)




    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100, vanilla_generation=False)
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,  # 这个是上下文
        target, # 这个是ground truth,用来计算得分的
        x,   # 这个是当前的prompt
        neighborhood=False
)-> typing.Dict:
    device =hparams.device
    if not isinstance(x, list):
        x = [x]
    if not isinstance(target, list):
        target = [target]

    if icl_examples is None:
        prompt = [f'Imagine that {item} ' for item in x]
        
    else:
        prompt = [''.join(icl_examples) + f'{item} ' for item in x]

    print("这是prompt")
    print(prompt)
    acc, contain_score, gen_content  = test_prediction_acc_LLM_judge(model, tokenizer, hparams, prompt, target, device, locality=False)
    return acc, contain_score, gen_content



    # if 't5' in model_name.lower():
    #     target_len = len(tokenizer.encode(target))
    #     target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
    #     encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
    #     input_ids = encodings['input_ids'].to(device)
    #     attention_mask = encodings['attention_mask'].to(device)
    #     with torch.no_grad():
    #         logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
    #         ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
    #         target_ids = target_ids[:,-target_len:-1]
    #         if neighborhood:
    #             return ans.squeeze().detach().cpu().numpy().tolist()
    #         return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    # elif 'llama' in model_name.lower():
    #     target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
    #     encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
    #     input_ids = encodings['input_ids'].to(device)
    #     attention_mask = encodings['attention_mask'].to(device)
    #     logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    #     ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
    #     target_ids = target_ids[:,1:]
    #     if neighborhood:
    #         return ans.squeeze().detach().cpu().numpy().tolist()
    #     return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    # else:
    #     target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
    #     encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
    #     input_ids = encodings['input_ids'].to(device)
    #     attention_mask = encodings['attention_mask'].to(device)
    #     logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    #     ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
    #     target_ids = target_ids[:,:-1]
    #     if neighborhood:
    #         return ans.squeeze().detach().cpu().numpy().tolist()
    #     return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
