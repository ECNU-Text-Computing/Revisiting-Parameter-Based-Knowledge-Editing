#!/usr/bin/env python
"""
Run SCR (Selective Context Retrieval) baseline experiments (RQ1).

SCR is a retrieval-based reference baseline that stores edited knowledge externally
and retrieves relevant facts at inference time, without modifying model parameters.

Usage:
    # Single edit
    python run_scr_baseline.py --model_path /path/to/llama3.1-8b \\
        --dataset_path /path/to/test_cf.json --memory_path /path/to/memory.json \\
        --top_k 5 --edit_scene single

    # Sequential editing (memory accumulates)
    python run_scr_baseline.py --model_path /path/to/llama3.1-8b \\
        --dataset_path /path/to/test_cf.json --memory_path /path/to/memory.json \\
        --top_k 5 --edit_scene sequential --memory_start 0 --memory_end 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scr'))

import argparse
from scr.edit_rag import edit, prepare_memory, load_data
from scr.prepare_requests import _prepare_requests


def parse_args():
    p = argparse.ArgumentParser(description='SCR retrieval-based editing baseline')
    p.add_argument('--model_path', required=True, help='Path to the target LLM')
    p.add_argument('--retriever_type', default='contriever-ms', help='Retriever type')
    p.add_argument('--retriever_path', required=True, help='Path to retriever model')
    p.add_argument('--dataset_path', required=True, help='Path to editing dataset (JSONL)')
    p.add_argument('--memory_path', required=True, help='Path to external memory (JSONL)')
    p.add_argument('--dataset_type', default='zsre', choices=['zsre', 'wiki_counterfact'],
                   help='Dataset type')
    p.add_argument('--top_k', type=int, default=5, help='Number of retrieved facts')
    p.add_argument('--eval_metric', default='contain', choices=['contain', 'exact_match', 'llm_judge'],
                   help='Evaluation metric')
    p.add_argument('--summary', action='store_true', default=True,
                   help='Use summary model for answer generation')
    p.add_argument('--edit_scene', default='sequential', choices=['single', 'sequential'],
                   help='Editing scenario')
    p.add_argument('--memory_start', type=int, default=0, help='Memory start index')
    p.add_argument('--memory_end', type=int, default=None, help='Memory end index')
    p.add_argument('--device', default='cuda:0', help='Device for model loading')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"SCR Baseline | Model: {args.model_path} | Top-K: {args.top_k} | "
          f"Scene: {args.edit_scene} | Eval: {args.eval_metric}")

    # Load memory
    memory = prepare_memory(
        args.memory_path,
        start_index=args.memory_start,
        end_index=args.memory_end,
    )

    # Load dataset and prepare requests
    requests = load_data(args.dataset_path, args.dataset_type)

    # Import SCR components
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    import torch

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map='auto'
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.pad_token_id = tok.eos_token_id

    retriever = AutoModel.from_pretrained(args.retriever_path).to(args.device)
    retriever_tok = AutoTokenizer.from_pretrained(args.retriever_path)

    # Build SCR tools
    from scr.edit_rag import RetrieveTool, SummaryTool, EditTool
    retrieve_tool = RetrieveTool(retriever, retriever_tok, args.device)
    summary_tool = SummaryTool(model, tok, args.device)
    edit_tool = EditTool(model, tok, args.device)

    # Prompt templates (from original SCR implementation)
    summary_prompt = (
        "Please identify the fact that best matches the core knowledge in the question. "
        "Only return the fact itself, no explanation."
    )
    answer_prompt = (
        "Based on the context, answer the question concisely."
    )

    # Run editing
    edit(
        retrieve_tool, args.top_k, summary_tool, edit_tool, memory, requests,
        summary_prompt, answer_prompt,
        eval_metric=args.eval_metric,
        summary=args.summary,
        edit_scene=args.edit_scene,
    )


if __name__ == '__main__':
    main()
