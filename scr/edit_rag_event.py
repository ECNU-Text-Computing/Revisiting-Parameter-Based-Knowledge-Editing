import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np
import random
import math
import os
import json
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import argparse
import re
from sentence_transformers import SentenceTransformer
import logging

# 设置日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
from prepare_requests import load_data, _prepare_requests
import time
def extract_fact(text):
    if text and not text.endswith('.'):
       text += '.'
    # Regex pattern to match the fact after the specified phrases, accounting for both quoted and non-quoted facts
    pattern = r'The fact that best matches the (question|core knowledge asked in the question|core knowledge in the question) is\s*[:\s]*["]?(.*?)(?=[".\n])'
    
    match = re.search(pattern, text)
    
    if match:
        fact = match.group(2).strip()  # Extract the fact part (group 2)
        
        # Ensure the fact includes a period if it doesn't end with one
        if fact and not fact.endswith('.'):
            fact += '.'
        
        return fact
    else:
        return text  # If no match found, return the original input text
        
def prepare_memory(datapath, num=None, start_index=None,end_index=None):
    memory = []
    with open(datapath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        for entry in data:
            sentence = entry["sentence"] if "sentence" in entry else entry["prompt"]
            memory.append(sentence)
    if start_index is not None and end_index is not None:
        memory = memory[start_index:end_index]
    if num is not None:
        memory = memory[:num]
    return memory

def llm_judge(question, ground_truth, prediction, api_key):
    content_template = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama's children.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha, Malia and Sasha, Malia and Sasha (repeated answer)
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target or contain repeated answer.


Here is a sample. Simply reply with either CORRECT or INCORRECT.

```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

According to the gold target, please grade the predicted answer of this question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
    """.strip()

    content = content_template.format(
        question=question,
        target=ground_truth,
        predicted_answer=prediction,
    )

    # client = OpenAI(
    #     api_key=api_key,
    # )
    client=  OpenAI(
            base_url=f"http://172.30.36.55:12345/v1",
            api_key="dummy"
        )

    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": ""},
    #         {"role": "user", "content": content}
    #     ],
    #     temperature=0.0
    # )

    completion = client.chat.completions.create(
        model="Qwen2.5-72-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        temperature=0.0
    )

    llm_ans = completion.choices[0].message.content
    llm_score = 1.0 if llm_ans == "A" else 0.0
    #time.sleep(1) # avoid high rate of request
    return llm_score

class RetrieveTool():
    def __init__(self, retriever_type, retriever_path, retriever, retriever_tok, device):
        if retriever_type == "ance":
            self.ance_model = SentenceTransformer(retriever_path)
        else:
            self.retriever = retriever
            self.retriever_tok = retriever_tok
        self.retriever_type=retriever_type
        self.device = device

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_sent_embeddings(self, memory, BSZ=1):
        if self.retriever_type == "ance":
            # Use ANCE model to encode sentences in memory
            self.memory_embedding = torch.tensor(self.ance_model.encode(memory)).to(self.device)
        else:
            all_embs = []
            for i in tqdm(range(0, len(memory), BSZ)):
                sent_batch = memory[i:i + BSZ]
                #start_time = time.time()   
                inputs = self.retriever_tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    outputs = self.retriever(**inputs)
                    embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
                all_embs.append(embeddings.cpu())
                #end_time = time.time()
                #elapsed_time = end_time - start_time
                # print("edit time")
                # with open("scr_edit_time.txt", "a") as f:    # 修改了这里
                #     f.write(f"Inference time: {elapsed_time:.4f} seconds\n")    # 修改了这里
        
            all_embs = torch.vstack(all_embs)
            self.memory_embedding = all_embs
        # self.memory_embedding = all_embs / all_embs.norm(dim=1, keepdim=True)  # L2 归一化

    def retrieve(self, query, memory, k):
        if self.retriever_type == "ance":
            # Use ANCE model to encode the query
            query_emb = torch.tensor(self.ance_model.encode([query])).to(self.device)
        else:
            inputs = self.retriever_tok([query], padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.retriever(**inputs)
                query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()

        # query_emb = query_emb / query_emb.norm()  # 对查询嵌入进行 L2 归一化

        sim = (query_emb @ self.memory_embedding.T)[0]
        knn = sim.topk(k, largest=True)
        fact_ids = knn.indices

        # Retrieve multiple facts
        retrieved_facts = [memory[fact_id] for fact_id in fact_ids]
        similarity_scores = knn.values.detach().cpu().numpy().tolist()

        return retrieved_facts, similarity_scores  # return a list


class SummaryTool():
    def __init__(self, summary_model, summary_tok, device):
        self.summary_model = summary_model
        self.summary_tok = summary_tok
        self.device = device

    def summary(self, question, retrieved_facts, summary_prompt, max_new_tokens=50):
        retrieved_facts = "\n".join(retrieved_facts)
        # 倒序排列
        # retrieved_facts = '\n'.join(retrieved_facts[::-1])
        
        prompt = summary_prompt.format(retrieved_facts=retrieved_facts, question=question)

        ids = self.summary_tok.encode(prompt, add_special_tokens=True)
        input_ids = torch.LongTensor([ids]).to(self.device)

        generated_ids = self.summary_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.summary_tok.eos_token_id,
            do_sample=False,

        )
        answer = self.summary_tok.decode(generated_ids[0].detach().cpu().numpy(), skip_special_tokens=True)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        ]
        relevance = self.summary_tok.decode(generated_ids[0].detach().cpu().numpy(), skip_special_tokens=True)
        # print("这是判断的结果")
        logging.debug(f"summary的结果: {relevance}")
        if "no relevant information" in relevance.lower() or "no relevant fact" in relevance.lower() or "false" in relevance.lower():
            return None
        else:
            relevance = extract_fact(relevance)
            #if "Explanation" in relevance:
            #    relevance = relevance.split("Explanation")[0].strip()
            return relevance
          

class EditTool():
    def __init__(self, model, tok, device):

        self.model = model
        self.tok = tok
        self.device = device

    def single_edit(self, prompt, target_new, knowledge=None, locality=False, vanilla_genration=True,
                    answer_prompt=None, eval_metric="token_em"):
        

        if eval_metric == "contain":
            if isinstance(target_new, str):
                target_new = [target_new]

            if knowledge is None:
                question = prompt
                #print(question)
                #target_new_tokens = self.tok.encode(target_new, add_special_tokens=False)
                prompt_tok = self.tok(
                    question,
                    return_tensors="pt",
                ).to(self.device)
                gen_token = self.model.generate(
                    input_ids=prompt_tok['input_ids'],
                    attention_mask=prompt_tok['attention_mask'],
                    max_new_tokens=50,
                    #min_new_tokens=30,  #################
                    stop_strings=[".","</s>", "<|endoftext|>"],   # 修改了这里
                    pad_token_id=self.tok.eos_token_id,
                    tokenizer=self.tok,
                    do_sample=False,
                    use_cache=True, 

                )
                out_answer_tokens = gen_token.detach().cpu().numpy().tolist()[0]

                input_length = prompt_tok['input_ids'].shape[1]  # 获取输入 token 的长度
                generated_tokens = gen_token.detach().cpu().numpy().tolist()[0][input_length:]  # 切片，获取生成的 token
                gen_content = self.tok.decode(generated_tokens, skip_special_tokens=True)
                suffixes_to_remove = [".", "\n", "</s>", "<|endoftext|>"]
                for suffix in suffixes_to_remove:
                    if gen_content.endswith(suffix):
                        gen_content = gen_content.rstrip(suffix)

                result = llm_judge(question, target_new[0],  gen_content.strip().split("\n")[0], "dummy")
                #max([float(target_new_answer.lower() in answer.lower()) for target_new_answer in target_new])

            else:
                question = prompt
                prompt = answer_prompt.format(question=question, knowledge=knowledge)
                #print(prompt)
                prompt_tok = self.tok(
                    prompt,
                    return_tensors="pt",
                ).to(self.device)
                gen_token = self.model.generate(
                    input_ids=prompt_tok['input_ids'],
                    attention_mask=prompt_tok['attention_mask'],
                    max_new_tokens=50,
                    #min_new_tokens=30,  #################
                    stop_strings=[".","</s>", "<|endoftext|>"],   # 修改了这里
                    tokenizer=self.tok,
                    pad_token_id=self.tok.eos_token_id,
                    do_sample=False,
                    use_cache=True, 

                )
 
                input_length = prompt_tok['input_ids'].shape[1]  # 获取输入 token 的长度
                generated_tokens = gen_token.detach().cpu().numpy().tolist()[0][input_length:]  # 切片，获取生成的 token
                gen_content = self.tok.decode(generated_tokens, skip_special_tokens=True)
                                
                suffixes_to_remove = [".", "\n", "</s>", "<|endoftext|>"]
                for suffix in suffixes_to_remove:
                    if gen_content.endswith(suffix):
                        gen_content = gen_content.rstrip(suffix)
                #result = max([float(target_new_answer.lower() in answer.lower()) for target_new_answer in target_new])
                result = llm_judge(question, target_new[0], gen_content.strip().split("\n")[0], "dummy")
            return gen_content,result


def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)

    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys


def summary_metrics_for_LLM_judge(all_metrics):
    if isinstance(all_metrics, dict):
        all_metrics = [all_metrics, ]
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # 加入时间戳
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(logs_dir, f'results_{time_str}.json')

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

    mean_metrics = dict()
    for eval in ["pre", "post"]:
        if eval not in all_metrics[0]:
            continue
        mean_metrics[eval] = dict()
        for key in ["rewrite_acc", "rephrase_acc", "rewrite_contain_acc", "rephrase_contain_acc", "fluency", 'rewrite_ppl', 'ood_acc']:
            if key in all_metrics[0][eval].keys():
                if key == "fluency":  # 对 fluency 字典值进行均值计算
                    fluency_values = [
                        metric[eval][key]['ngram_entropy'] if 'ngram_entropy' in metric[eval][key] else np.nan
                        for metric in all_metrics if key in metric[eval]
                    ]
                    fluency_values = [val for val in fluency_values if not np.isnan(val)]  # 移除无效值
                    mean_metrics[eval][key] = np.mean(fluency_values) if fluency_values else np.nan
                else:  # 对其他评估指标进行均值计算
                    mean_metrics[eval][key] = np.mean(
                        [np.mean(metric[eval][key]) for metric in all_metrics if key in metric[eval]]
                    )
        for key in ["locality", "portability"]:
            if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                mean_metrics[eval][key] = dict()
                sub_keys = set()
                for metric in all_metrics:
                    if key in metric[eval]:
                        sub_keys.update(metric[eval][key].keys())

                for sub_key in sub_keys:
                    if not sub_key.endswith("_acc"):
                        continue

                    per_sample_means = []
                    for metric in all_metrics:
                        if key in metric[eval] and sub_key in metric[eval][key]:
                            vals = metric[eval][key][sub_key]
                            if isinstance(vals, list):
                                if len(vals) == 0:
                                    continue
                                # 若是 list of list，flatten
                                if isinstance(vals[0], list):
                                    vals = [v for v in vals if isinstance(v, list) and len(v) > 0]
                                    vals = [item for sublist in vals for item in sublist]
                                vals = [v for v in vals if isinstance(v, (float, int))]
                                if len(vals) > 0:
                                    per_sample_means.append(np.mean(vals))

                    if len(per_sample_means) > 0:
                        mean_metrics[eval][key][sub_key] = np.mean(per_sample_means)


    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
    print("Metrics Summary: ", mean_metrics)


def edit(retrieve_tool, k, summary_tool, edit_tool, memory, requests, summary_prompt, answer_prompt,
         eval_metric,summary=True,edit_scene="sequential"):
    if edit_scene=="sequential":
        all_metrics = []
        # 得到所有memory的embedding表示
        retrieve_tool.get_sent_embeddings(memory, BSZ=32)
        score_list = []
        for i, request in enumerate(tqdm(requests, total=len(requests))):
            ret = {}
            ret = request
            ret["post"]={}

            if "portability" in request and any(request["portability"]):
                ret["post"]["portability"] = {}
                ret_portability = {}
            
                for portability_key in request['portability'].keys():
                    ret_portability[f"{portability_key}_gen_content"] = []
                    ret_portability[f"{portability_key}_acc"] = []
                    #ret_portability[f"{portability_key}_summary_acc"] = []
                    for j in range(len(request['portability'][portability_key]["prompt"])):
                        question = request['portability'][portability_key]["prompt"][j]
                        target_new = request['portability'][portability_key]["ground_truth"][j]

                        # retrieve
                        retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)
                
                        # summary
                        if summary:
                            related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)
                        else:
                            related_knowledge = retrieved_facts[0]

                
                        # edit
                        gen_content,portability_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                                locality=False, vanilla_genration=True,
                                                                answer_prompt=answer_prompt,eval_metric=eval_metric)
                        ret_portability[f"{portability_key}_gen_content"].append(gen_content)
                        ret_portability[f"{portability_key}_acc"].append(portability_acc)
                        #ret_portability[f"{portability_key}_summary_acc"].append(portability_summary_acc)
                ret["post"]["portability"] = ret_portability

            if "locality" in request and any(request["locality"]):

                ret["post"]['locality'] = {}
                ret_locality = {}
                for locality_key in request['locality'].keys():
                    ret_locality[f"{locality_key}_gen_content"]=[]
                    ret_locality[f"{locality_key}_acc"] = []
                    #ret['locality'][f"{locality_key}_summary_acc"] = []
                    for j in range(len(request['locality'][locality_key]["prompt"])):
                        question = request['locality'][locality_key]["prompt"][j]
                        target_new = request['locality'][locality_key]["ground_truth"][j]

                        # retrieve
                        retrieved_facts, sim_score = retrieve_tool.retrieve(question, memory, k)
                        # summary
                        if summary:
                            related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)
                        else:
                            related_knowledge = retrieved_facts[0]
                
                
                        # edit
                        gen_content,locality_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                                locality=True, vanilla_genration=True,
                                                                answer_prompt=answer_prompt,eval_metric=eval_metric)
                        ret_locality[f"{locality_key}_gen_content"].append(gen_content)
                        ret_locality[f"{locality_key}_acc"].append(locality_acc)
                        #ret['locality'][f"{locality_key}_summary_acc"].append(locality_summary_acc)
                ret["post"]["locality"]=ret_locality
            all_metrics.append(ret)
        print(all_metrics)

        summary_metrics_for_LLM_judge(all_metrics)
    else:
        all_metrics = []
        
        score_list = []
        for i, request in enumerate(tqdm(requests, total=len(requests))):

            ret = {}
            ret = request
            ret["post"]={}


            if "portability" in request and any(request["portability"]):
                ret["post"]["portability"] = {}
                ret_portability = {}
            
                for portability_key in request['portability'].keys():
                    ret_portability[f"{portability_key}_gen_content"] = []
                    ret_portability[f"{portability_key}_acc"] = []
                    #ret_portability[f"{portability_key}_summary_acc"] = []
                    for j in range(len(request['portability'][portability_key]["prompt"])):
                        question = request['portability'][portability_key]["prompt"][j]
                        target_new = request['portability'][portability_key]["ground_truth"][j]

                        # retrieve
                        retrieved_facts=[memory[i]]
                        related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)
                        # edit
                        gen_content,portability_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                                locality=False, vanilla_genration=True,
                                                                answer_prompt=answer_prompt,eval_metric=eval_metric)
                        ret_portability[f"{portability_key}_gen_content"].append(gen_content)
                        ret_portability[f"{portability_key}_acc"].append(portability_acc)
                        #ret_portability[f"{portability_key}_summary_acc"].append(portability_summary_acc)
                ret["post"]["portability"] = ret_portability

            if "locality" in request and any(request["locality"]):

                ret["post"]['locality'] = {}
                ret_locality = {}
                for locality_key in request['locality'].keys():
                    ret_locality[f"{locality_key}_gen_content"]=[]
                    ret_locality[f"{locality_key}_acc"] = []
                    #ret['locality'][f"{locality_key}_summary_acc"] = []
                    for j in range(len(request['locality'][locality_key]["prompt"])):
                        question = request['locality'][locality_key]["prompt"][j]
                        target_new = request['locality'][locality_key]["ground_truth"][j]

                        # retrieve
                        retrieved_facts=[memory[i]]
                        # summary

                        related_knowledge = summary_tool.summary(question, retrieved_facts, summary_prompt, max_new_tokens=50)

                        # edit
                        gen_content,locality_acc = edit_tool.single_edit(question, target_new, knowledge=related_knowledge,
                                                                locality=True, vanilla_genration=True,
                                                                answer_prompt=answer_prompt,eval_metric=eval_metric)
                        ret_locality[f"{locality_key}_gen_content"].append(gen_content)
                        ret_locality[f"{locality_key}_acc"].append(locality_acc)
                        #ret['locality'][f"{locality_key}_summary_acc"].append(locality_summary_acc)
                ret["post"]["locality"]=ret_locality
            all_metrics.append(ret)
        print(all_metrics)

        summary_metrics_for_LLM_judge(all_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 显式指定路径
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--retriever_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--memory_path', type=str, default=None)

    parser.add_argument('--retriever_type', type=str, default="contriever-ms")
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--ds_size', type=int, default=None)
    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--end_index', type=int, default=None)

    parser.add_argument('--memory_size', type=int, default=None)
    parser.add_argument('--memory_start_index', type=int, default=None)
    parser.add_argument('--memory_end_index', type=int, default=None)

    parser.add_argument('--eval_metric', type=str, default="contain", choices=['token_em', 'contain'])
    parser.add_argument('--summary', action='store_true', default=False)
    parser.add_argument('--edit_scene', type=str, default=None, choices=['single', 'sequential'])

    args = parser.parse_args()

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    summary_model = model
    summary_tok = tok

    # 加载检索器
    retriever = AutoModel.from_pretrained(args.retriever_path).to(args.device)
    retriever_tok = AutoTokenizer.from_pretrained(args.retriever_path)

    # 加载 memory（如提供）
    memory = None
    if args.memory_path:
        memory = prepare_memory(
            args.memory_path,
            num=args.memory_size,
            start_index=args.memory_start_index,
            end_index=args.memory_end_index
        )

    # 加载数据
    data = load_data(args.data_path, eval_metric=args.eval_metric)

    # 构造请求
    if args.ds_size is None:
        requests = _prepare_requests(
            prompts=data["prompts"],
            target_new=data["target_new"],
            locality_inputs=data.get("locality_inputs"),
            portability_inputs=data.get("portability_inputs")
        )
    else:
        requests = _prepare_requests(
            prompts=data["prompts"],
            target_new=data["target_new"],
            locality_inputs=data.get("locality_inputs"),
            portability_inputs=data.get("portability_inputs")
        )[args.start_index:args.end_index]

    # Prompt
    if args.top_k==1:
        summary_prompt = """Given a set of facts and a question, return the fact that best matches the core knowledge asked in the question. If the question cannot be answered with the facts, return "no relevant fact."\n\nFacts: {retrieved_facts}\nQuestion: {question}\n\nOutput:"""
    else:
        summary_prompt = """Given a set of facts and a question, return the fact (the entire sentence) that best matches the core knowledge asked in the question. If the question cannot be answered with the facts, return "no relevant fact."\n\nFacts: \n{retrieved_facts}\nQuestion: {question}\n\nOutput:"""
    answer_prompt = """Answer the question based on the Fact.\n\nFact: {knowledge}\nQuestion: {question}\nAnswer:"""


    retrieve_tool = RetrieveTool(args.retriever_type, args.retriever_path, retriever, retriever_tok, args.device)
    summary_tool = SummaryTool(summary_model, summary_tok, args.device)
    edit_tool = EditTool(model, tok, args.device)


    edit(
        retrieve_tool,
        args.top_k,
        summary_tool,
        edit_tool,
        memory,
        requests,
        summary_prompt,
        answer_prompt,
        eval_metric=args.eval_metric,
        summary=args.summary,
        edit_scene=args.edit_scene
    )



