#!/usr/bin/env python
"""
Run knowledge editing on reasoning-oriented LLMs (RQ2).

Evaluates whether parameter-based editing methods can integrate new facts into
reasoning LLMs (e.g., DeepSeek-R1-Distill-Llama-8B) without degrading their
mathematical reasoning capabilities.

After editing, evaluates on math reasoning benchmarks (GSM8K, MATH, etc.)
to quantify reasoning collapse.

Usage:
    python run_reasoning_edit.py --method AlphaEdit --model deepseek-r1 --dataset zsre \\
        --N 100 --sequential --reasoning_bench gsm8k
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyeditor import (
    FTHyperParams, IKEHyperParams, MEMITHyperParams, ROMEHyperParams,
    LoRAHyperParams, MENDHyperParams, GraceHyperParams, WISEHyperParams,
    PMETHyperParams, AlphaEditHyperParams,
)
from easyeditor import BaseEditor
from easyeditor import KnowEditDataset

METHOD_HPARAMS = {
    'ROME': ROMEHyperParams, 'MEMIT': MEMITHyperParams,
    'PMET': PMETHyperParams, 'AlphaEdit': AlphaEditHyperParams,
    'FT': FTHyperParams, 'MEND': MENDHyperParams,
    'WISE': WISEHyperParams, 'LoRA': LoRAHyperParams,
    'GRACE': GraceHyperParams, 'IKE': IKEHyperParams,
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Knowledge editing on reasoning-oriented LLMs (RQ2)'
    )
    p.add_argument('--method', required=True, choices=list(METHOD_HPARAMS.keys()))
    p.add_argument('--model', required=True, help='Reasoning LLM path or name')
    p.add_argument('--config_path', required=True, help='Path to hyperparameter YAML')
    p.add_argument('--data_path', required=True, help='Path to editing dataset')
    p.add_argument('--N', type=int, default=100, help='Number of edits')
    p.add_argument('--sequential', action='store_true', default=True)
    p.add_argument('--reasoning_bench', nargs='+', default=['gsm8k'],
                   help='Math reasoning benchmarks to evaluate')
    p.add_argument('--eval_mode', default='llm_judge',
                   choices=['exact_match', 'llm_judge'])
    p.add_argument('--output_dir', default='outputs')
    return p.parse_args()


def evaluate_reasoning(model, tok, benchmark_name, device='cuda:0'):
    """
    Evaluate edited model on a mathematical reasoning benchmark.
    Returns accuracy score.
    """
    # Placeholder — users should plug in their benchmark evaluation code.
    # Common benchmarks: GSM8K, MATH, GPQA-Diamond, ARC-c, MMLU-Pro
    print(f"  Evaluating reasoning on: {benchmark_name}")
    # TODO: integrate dataset-specific evaluation
    return {'benchmark': benchmark_name, 'accuracy': None}


def main():
    args = parse_args()
    print(f"RQ2: Editing {args.method} on {args.model} | N={args.N} | "
          f"Sequential={args.sequential}")

    # Load and configure editor
    hparams_cls = METHOD_HPARAMS[args.method]
    hparams = hparams_cls.from_hparams(args.config_path)
    hparams.evaluation_type = args.eval_mode
    editor = BaseEditor.from_hparams(hparams)

    # Load data
    dataset = KnowEditDataset(args.data_path, size=args.N)
    prompts = [d['prompt'] for d in dataset]
    subjects = [d['subject'] for d in dataset]
    target_new = [d['target_new'] for d in dataset]
    rephrase_prompts = [d['rephrase_prompt'] for d in dataset]

    # Run editing
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subjects,
        sequential_edit=args.sequential,
    )

    # Evaluate reasoning benchmarks
    print("\n=== Reasoning Benchmark Evaluation ===")
    reasoning_results = {}
    for bench in args.reasoning_bench:
        result = evaluate_reasoning(edited_model, editor.tok, bench)
        reasoning_results[bench] = result
        print(f"  {bench}: {result}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output = {
        'editing_metrics': metrics,
        'reasoning_results': reasoning_results,
        'config': {
            'method': args.method, 'model': args.model,
            'N': args.N, 'sequential': args.sequential,
        },
    }
    fname = f'reasoning_{args.method}_{args.model}_N{args.N}_{timestamp}.json'
    with open(os.path.join(args.output_dir, fname), 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to: {os.path.join(args.output_dir, fname)}")


if __name__ == '__main__':
    main()
