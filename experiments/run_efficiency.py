#!/usr/bin/env python
"""
Measure editing time and inference latency across methods (RQ4).

Evaluates the trade-off between:
  - Edit time: wall-clock time to apply a single knowledge edit
  - Inference time: average latency per query normalized by the base model

Usage:
    python run_efficiency.py --method ROME --model llama3.1-8b --config_path configs/ROME/llama3.1-8b.yaml
"""

import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyeditor import (
    FTHyperParams, IKEHyperParams, MEMITHyperParams, ROMEHyperParams,
    LoRAHyperParams, MENDHyperParams, GraceHyperParams, WISEHyperParams,
    PMETHyperParams, AlphaEditHyperParams,
)
from easyeditor import BaseEditor

METHOD_HPARAMS = {
    'ROME': ROMEHyperParams, 'MEMIT': MEMITHyperParams,
    'PMET': PMETHyperParams, 'AlphaEdit': AlphaEditHyperParams,
    'FT': FTHyperParams, 'MEND': MENDHyperParams,
    'WISE': WISEHyperParams, 'LoRA': LoRAHyperParams,
    'GRACE': GraceHyperParams, 'IKE': IKEHyperParams,
}


def parse_args():
    p = argparse.ArgumentParser(description='Efficiency benchmarking (RQ4)')
    p.add_argument('--method', required=True, choices=list(METHOD_HPARAMS.keys()))
    p.add_argument('--model', required=True)
    p.add_argument('--config_path', required=True)
    p.add_argument('--num_edits', type=int, default=10, help='Number of edits to measure')
    p.add_argument('--num_inference', type=int, default=50, help='Queries for inference timing')
    p.add_argument('--output_len', type=int, default=50, help='Output token length')
    p.add_argument('--output_dir', default='outputs')
    return p.parse_args()


def measure_edit_time(editor, prompts, target_new, subjects):
    """Measure wall-clock time for a single edit."""
    start = time.time()
    editor.edit(
        prompts=[prompts[0]],
        target_new=[target_new[0]],
        subject=[subjects[0]],
        sequential_edit=False,
    )
    return time.time() - start


def measure_inference_latency(model, tok, queries, output_len, device):
    """Measure average inference latency per query."""
    import torch
    latencies = []
    for query in queries:
        inputs = tok(query, return_tensors='pt').to(device)
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=output_len, do_sample=False)
        latencies.append(time.time() - start)
    return np.mean(latencies)


def main():
    args = parse_args()

    # Build editor
    hparams_cls = METHOD_HPARAMS[args.method]
    hparams = hparams_cls.from_hparams(args.config_path)

    # --- Measure base model inference ---
    editor_base = BaseEditor.from_hparams(hparams)
    base_model = editor_base.model
    base_tok = editor_base.tok
    device = f'cuda:{hparams.device}'

    sample_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When did World War II end?",
    ] * (args.num_inference // 4 + 1)
    sample_queries = sample_queries[:args.num_inference]

    base_latency = measure_inference_latency(
        base_model, base_tok, sample_queries, args.output_len, device
    )
    print(f"Base model avg latency ({args.num_inference} queries): {base_latency:.4f}s")

    # --- Measure edit time ---
    sample_prompts = ["Who is the president of the United States?"] * args.num_edits
    sample_targets = ["Joe Biden"] * args.num_edits
    sample_subjects = ["president of the United States"] * args.num_edits

    edit_times = []
    for i in range(args.num_edits):
        et = measure_edit_time(
            editor_base, [sample_prompts[i]], [sample_targets[i]], [sample_subjects[i]]
        )
        edit_times.append(et)

    avg_edit_time = np.mean(edit_times)
    print(f"Avg edit time ({args.num_edits} edits): {avg_edit_time:.4f}s")

    # --- Measure post-edit inference ---
    post_latency = measure_inference_latency(
        base_model, base_tok, sample_queries, args.output_len, device
    )
    normalized_latency = post_latency / base_latency if base_latency > 0 else 1.0
    print(f"Post-edit avg latency: {post_latency:.4f}s (normalized: {normalized_latency:.3f}x)")

    # Save results
    results = {
        'method': args.method,
        'model': args.model,
        'avg_edit_time_s': avg_edit_time,
        'base_inference_latency_s': base_latency,
        'post_edit_inference_latency_s': post_latency,
        'normalized_latency': normalized_latency,
        'num_edits_measured': args.num_edits,
        'num_inference_queries': args.num_inference,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    fname = f'efficiency_{args.method}_{args.model}.json'
    with open(os.path.join(args.output_dir, fname), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {os.path.join(args.output_dir, fname)}")


if __name__ == '__main__':
    main()
