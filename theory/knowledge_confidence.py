import json
import torch
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
def load_model_and_tokenizer(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"模型加载完成，设备: {device}")
    return model, tokenizer, device

# 读取数据集 - 修改为加载完整原始数据
def load_original_dataset(file_path: str) -> List[Dict]:
    """从JSON文件中加载完整原始数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"从 {file_path} 加载了 {len(data)} 个样本")
    return data

# 从数据中提取QA对
def extract_qa_pairs(data: List[Dict]) -> List[Tuple[int, str, str]]:
    """提取索引、问题和答案"""
    qa_pairs = []
    for idx, item in enumerate(data):
        question = item.get("prompt", "")
        # ground_truth 是列表，取第一个元素
        answer_list = item.get("ground_truth", [])
        answer = answer_list[0] if answer_list else ""
        
        if question and answer:
            qa_pairs.append((idx, question, answer))
    
    print(f"提取了 {len(qa_pairs)} 个有效的QA对")
    return qa_pairs

# 构建提示词
def build_prompt(question: str, answer: str, tokenizer) -> Tuple[str, str]:
    """
    返回两个prompt:
    - prompt_qa: 完整的问答，用于计算logprob
    - prompt_q: 只有问题，用于确定问题长度
    """
    # 使用Llama风格的格式
    prompt_qa = f"Question: {question}\nAnswer: {answer}"
    prompt_q = f"Question: {question}\nAnswer:"
    
    return prompt_qa, prompt_q

# 计算答案的log概率
@torch.no_grad()
def compute_answer_logprob(
    question: str,
    answer: str,
    tokenizer,
    model,
    device: str,
) -> dict:
    """
    计算answer部分token的平均log-probability
    """
    prompt_qa, prompt_q = build_prompt(question, answer, tokenizer)
    
    # Tokenize
    ids_qa = tokenizer(prompt_qa, return_tensors="pt").input_ids.to(device)
    ids_q = tokenizer(prompt_q, return_tensors="pt").input_ids.to(device)
    
    len_q = ids_q.shape[1]
    len_qa = ids_qa.shape[1]
    answer_len = len_qa - len_q
    
    if answer_len <= 0:
        return {"avg_logprob": float("-inf"), "token_count": 0, "per_token_logprobs": []}
    
    # 前向传播
    with torch.no_grad():
        outputs = model(ids_qa)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # log_softmax → log-prob
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [1, seq_len, vocab]
    
    # answer token的logprob
    answer_token_ids = ids_qa[0, len_q:]  # shape: [answer_len]
    answer_logits_pos = log_probs[0, len_q-1:len_qa-1, :]  # shape: [answer_len, vocab]
    
    per_token_lp = answer_logits_pos[
        torch.arange(answer_len, device=answer_logits_pos.device), answer_token_ids
    ].cpu().tolist()  # list of float
    
    avg_lp = sum(per_token_lp) / len(per_token_lp) if per_token_lp else float("-inf")
    
    return {
        "avg_logprob": avg_lp,
        "token_count": answer_len,
        "per_token_logprobs": per_token_lp,
    }

# 主处理函数
def process_dataset(
    file_path: str,
    model_path: str,
    batch_size: int = 8
) -> Tuple[List, List, List]:
    """
    处理整个数据集，按置信度划分为高、中、低三类
    返回：高置信度组、中置信度组、低置信度组（每个组都是完整的原始数据样本）
    """
    # 1. 加载完整原始数据
    original_data = load_original_dataset(file_path)
    qa_pairs = extract_qa_pairs(original_data)
    total_items = len(qa_pairs)
    print(f"总共 {total_items} 个QA对需要处理")
    
    # 2. 加载模型
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
    # 3. 计算所有QA对的置信度
    confidence_items = []
    
    for i, (idx, question, answer) in enumerate(qa_pairs):
        if (i + 1) % 10 == 0 or i == 0 or i == total_items - 1:
            print(f"处理进度: {i+1}/{total_items} ({(i+1)/total_items*100:.1f}%)")
        
        try:
            result = compute_answer_logprob(question, answer, tokenizer, model, device)
            confidence_items.append({
                "original_index": idx,
                "avg_logprob": result["avg_logprob"],
                "token_count": result["token_count"]
            })
        except Exception as e:
            print(f"处理第 {i} 个样本时出错: {e}")
            confidence_items.append({
                "original_index": idx,
                "avg_logprob": float("-inf"),
                "token_count": 0
            })
    
    # 4. 按置信度排序（logprob越高，置信度越高）
    confidence_items.sort(key=lambda x: x["avg_logprob"], reverse=True)
    
    # 5. 平均分为3类
    # n = len(confidence_items)
    # group_size = n // 3
    
    # # 高置信度组索引
    # high_indices = [item["original_index"] for item in confidence_items[:group_size]]
    # high_confidence_data = []
    # for idx in high_indices:
    #     sample = original_data[idx].copy()
    #     # 找到对应的置信度信息
    #     conf_item = next((item for item in confidence_items if item["original_index"] == idx), None)
    #     if conf_item:
    #         sample["avg_logprob"] = conf_item["avg_logprob"]
    #     high_confidence_data.append(sample)
    
    # # 中置信度组索引
    # mid_indices = [item["original_index"] for item in confidence_items[group_size:2*group_size]]
    # mid_confidence_data = []
    # for idx in mid_indices:
    #     sample = original_data[idx].copy()
    #     conf_item = next((item for item in confidence_items if item["original_index"] == idx), None)
    #     if conf_item:
    #         sample["avg_logprob"] = conf_item["avg_logprob"]
    #     mid_confidence_data.append(sample)
    
    # # 低置信度组索引
    # low_indices = [item["original_index"] for item in confidence_items[2*group_size:]]
    # low_confidence_data = []
    # for idx in low_indices:
    #     sample = original_data[idx].copy()
    #     conf_item = next((item for item in confidence_items if item["original_index"] == idx), None)
    #     if conf_item:
    #         sample["avg_logprob"] = conf_item["avg_logprob"]
    #     low_confidence_data.append(sample)

    # base_size = n // 3
    # remainder = n % 3

    # # 计算各组分界点
    # if remainder == 0:
    #     # 正好整除
    #     split1 = base_size
    #     split2 = 2 * base_size
    # elif remainder == 1:
    #     # 余1，给中间组
    #     split1 = base_size
    #     split2 = 2 * base_size + 1
    # else:  # remainder == 2
    #     # 余2，高低组各多1个
    #     split1 = base_size + 1
    #     split2 = split1 + base_size

    # # 高置信度组
    # high_indices = [item["original_index"] for item in confidence_items[:split1]]
    # high_confidence_data = [
    #     {**original_data[idx].copy(), "avg_logprob": conf_item["avg_logprob"]}
    #     for idx, conf_item in [(idx, next((item for item in confidence_items if item["original_index"] == idx))) 
    #                         for idx in high_indices]
    # ]

    # # 中置信度组
    # mid_indices = [item["original_index"] for item in confidence_items[split1:split2]]
    # mid_confidence_data = [
    #     {**original_data[idx].copy(), "avg_logprob": conf_item["avg_logprob"]}
    #     for idx, conf_item in [(idx, next((item for item in confidence_items if item["original_index"] == idx))) 
    #                         for idx in mid_indices]
    # ]

    # # 低置信度组
    # low_indices = [item["original_index"] for item in confidence_items[split2:]]
    # low_confidence_data = [
    #     {**original_data[idx].copy(), "avg_logprob": conf_item["avg_logprob"]}
    #     for idx, conf_item in [(idx, next((item for item in confidence_items if item["original_index"] == idx))) 
    #                         for idx in low_indices]
    # ]
    
    # # 打印统计信息
    # print("\n=== 置信度分组统计 ===")
    # print(f"高置信度组: {len(high_confidence_data)} 个样本")
    # high_logprobs = [x.get("avg_logprob", float("-inf")) for x in high_confidence_data]
    # print(f"  - 平均logprob: {np.mean(high_logprobs):.4f}")
    # print(f"  - 范围: {max(high_logprobs):.4f} 到 {min(high_logprobs):.4f}")
    
    # print(f"中置信度组: {len(mid_confidence_data)} 个样本")
    # mid_logprobs = [x.get("avg_logprob", float("-inf")) for x in mid_confidence_data]
    # print(f"  - 平均logprob: {np.mean(mid_logprobs):.4f}")
    # print(f"  - 范围: {max(mid_logprobs):.4f} 到 {min(mid_logprobs):.4f}")
    
    # print(f"低置信度组: {len(low_confidence_data)} 个样本")
    # low_logprobs = [x.get("avg_logprob", float("-inf")) for x in low_confidence_data]
    # print(f"  - 平均logprob: {np.mean(low_logprobs):.4f}")
    # print(f"  - 范围: {max(low_logprobs):.4f} 到 {min(low_logprobs):.4f}")
    
    # return high_confidence_data, mid_confidence_data, low_confidence_data

    # 5. 按设定数量划分数据集
    n = len(confidence_items)

    # 检查样本数量是否足够
    if n < 200:
        print(f"警告: 样本总数({n})不足200，无法满足高/低置信度各100个的要求")
        print("将按比例调整划分策略...")
        
        # 当样本不足时，按比例分配
        high_count = min(100, n // 3)
        low_count = min(100, n // 3)
        mid_count = n - high_count - low_count
        
        # 确保mid_count不为负数
        if mid_count < 0:
            low_count = max(0, low_count + mid_count)  # 调整低置信度数量
            mid_count = n - high_count - low_count
    else:
        # 正常情况：高/低各100，其余为中
        high_count = 100
        low_count = 100
        mid_count = n - high_count - low_count

    # 高置信度组（前100个）
    high_indices = [item["original_index"] for item in confidence_items[:high_count]]
    high_confidence_data = [
        {**original_data[idx].copy(), "avg_logprob": item["avg_logprob"]}
        for item in confidence_items[:high_count]
        for idx in [item["original_index"]]
    ]

    # 低置信度组（最后100个）
    low_indices = [item["original_index"] for item in confidence_items[-low_count:]]
    low_confidence_data = [
        {**original_data[idx].copy(), "avg_logprob": item["avg_logprob"]}
        for item in confidence_items[-low_count:]
        for idx in [item["original_index"]]
    ]

    # 中置信度组（中间部分）
    mid_indices = [item["original_index"] for item in confidence_items[high_count:high_count+mid_count]]
    mid_confidence_data = [
        {**original_data[idx].copy(), "avg_logprob": item["avg_logprob"]}
        for item in confidence_items[high_count:high_count+mid_count]
        for idx in [item["original_index"]]
    ]

    # 打印统计信息
    print("\n=== 置信度分组统计 ===")
    print(f"高置信度组: {len(high_confidence_data)} 个样本")
    high_logprobs = [x.get("avg_logprob", float("-inf")) for x in high_confidence_data]
    if high_logprobs:
        print(f"  - 平均logprob: {np.mean(high_logprobs):.4f}")
        print(f"  - 范围: {max(high_logprobs):.4f} 到 {min(high_logprobs):.4f}")

    print(f"中置信度组: {len(mid_confidence_data)} 个样本")
    mid_logprobs = [x.get("avg_logprob", float("-inf")) for x in mid_confidence_data]
    if mid_logprobs:
        print(f"  - 平均logprob: {np.mean(mid_logprobs):.4f}")
        print(f"  - 范围: {max(mid_logprobs):.4f} 到 {min(mid_logprobs):.4f}")

    print(f"低置信度组: {len(low_confidence_data)} 个样本")
    low_logprobs = [x.get("avg_logprob", float("-inf")) for x in low_confidence_data]
    if low_logprobs:
        print(f"  - 平均logprob: {np.mean(low_logprobs):.4f}")
        print(f"  - 范围: {max(low_logprobs):.4f} 到 {min(low_logprobs):.4f}")

    # 验证分组正确性
    print(f"\n=== 分组验证 ===")
    print(f"总样本数: {n}")
    print(f"分组样本数之和: {len(high_confidence_data) + len(mid_confidence_data) + len(low_confidence_data)}")
    if len(high_confidence_data) + len(mid_confidence_data) + len(low_confidence_data) == n:
        print("分组验证通过!")
    else:
        print("警告: 分组样本数与总数不一致!")

    return high_confidence_data, mid_confidence_data, low_confidence_data

# 保存结果
def save_results(high_data: List, mid_data: List, low_data: List, output_dir: str = "./confidence_groups"):
    """保存结果到文件，保持原始数据格式"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON
    with open(f"{output_dir}/high_confidence.json", "w", encoding="utf-8") as f:
        json.dump(high_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/mid_confidence.json", "w", encoding="utf-8") as f:
        json.dump(mid_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/low_confidence.json", "w", encoding="utf-8") as f:
        json.dump(low_data, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats = {
        "total_samples": len(high_data) + len(mid_data) + len(low_data),
        "high_confidence": {
            "count": len(high_data),
            "avg_logprob": np.mean([x.get("avg_logprob", 0) for x in high_data]) if high_data else 0
        },
        "mid_confidence": {
            "count": len(mid_data),
            "avg_logprob": np.mean([x.get("avg_logprob", 0) for x in mid_data]) if mid_data else 0
        },
        "low_confidence": {
            "count": len(low_data),
            "avg_logprob": np.mean([x.get("avg_logprob", 0) for x in low_data]) if low_data else 0
        }
    }
    
    with open(f"{output_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {output_dir}/ 目录")

# 主函数
def main():
    # 设置路径
    file_path = "/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/zsre/ZsRE-test-all.json"
    model_path = "/home/wyren/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    output_dir = "/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/empirical_exp/results/knowledge_confidence"

    # 处理数据集
    print("开始处理数据集...")
    high_data, mid_data, low_data = process_dataset(file_path, model_path)
    
    # 保存结果
    save_results(high_data, mid_data, low_data, output_dir)
    
    # 显示一些示例
    print("\n=== 示例样本 ===")
    print("高置信度组示例:")
    for i in range(min(2, len(high_data))):
        sample = high_data[i]
        print(f"  {i+1}. Q: {sample.get('prompt', '')[:50]}...")
        print(f"     A: {sample.get('ground_truth', [''])[0]}, logprob: {sample.get('avg_logprob', 0):.4f}")
    
    print("\n中置信度组示例:")
    for i in range(min(2, len(mid_data))):
        sample = mid_data[i]
        print(f"  {i+1}. Q: {sample.get('prompt', '')[:50]}...")
        print(f"     A: {sample.get('ground_truth', [''])[0]}, logprob: {sample.get('avg_logprob', 0):.4f}")
    
    print("\n低置信度组示例:")
    for i in range(min(2, len(low_data))):
        sample = low_data[i]
        print(f"  {i+1}. Q: {sample.get('prompt', '')[:50]}...")
        print(f"     A: {sample.get('ground_truth', [''])[0]}, logprob: {sample.get('avg_logprob', 0):.4f}")
    
    return high_data, mid_data, low_data

if __name__ == "__main__":
    # 执行主函数
    result_high, result_mid, result_low = main()
    
    # 返回结果
    print("\n处理完成！")
    print(f"高置信度样本数: {len(result_high)}")
    print(f"中置信度样本数: {len(result_mid)}")
    print(f"低置信度样本数: {len(result_low)}")