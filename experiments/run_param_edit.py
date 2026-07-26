#!/usr/bin/env python
"""
Run parameter-based knowledge editing experiments (RQ1).

Supports single and sequential editing across all parameter-modifying methods
evaluated in the paper: ROME, MEMIT, PMET, AlphaEdit, FT, MEND, WISE, LoRA, GRACE, IKE.

Usage:
    # Single edit with ROME on ZsRE
    python run_param_edit.py --method ROME --model llama3.1-8b --dataset zsre --N 1

    # Sequential editing (N=100) with AlphaEdit on WikiData-counterfact
    python run_param_edit.py --method AlphaEdit --model llama3.1-8b --dataset wiki_counterfact --N 100 --sequential

    # Full dataset with MEND on Mistral-7B
    python run_param_edit.py --method MEND --model mistral-7b --dataset zsre --sequential

    # LLM-judge evaluation (semantic consistency via Qwen2.5-72B)
    python run_param_edit.py --method ROME --model llama2-7b --dataset zsre --N 100 --sequential --eval_mode llm_judge
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyeditor import (
    FTHyperParams, IKEHyperParams, MEMITHyperParams, ROMEHyperParams,
    LoRAHyperParams, MENDHyperParams, GraceHyperParams, WISEHyperParams,
    PMETHyperParams, AlphaEditHyperParams,
)
from easyeditor import BaseEditor
from easyeditor import ZsreDataset, WikiCounterfactDataset, KnowEditDataset
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Supported method → hyperparameter class mapping
# ---------------------------------------------------------------------------
METHOD_HPARAMS = {
    'ROME': ROMEHyperParams,
    'MEMIT': MEMITHyperParams,
    'PMET': PMETHyperParams,
    'AlphaEdit': AlphaEditHyperParams,
    'FT': FTHyperParams,
    'MEND': MENDHyperParams,
    'WISE': WISEHyperParams,
    'LoRA': LoRAHyperParams,
    'GRACE': GraceHyperParams,
    'IKE': IKEHyperParams,
}

# ---------------------------------------------------------------------------
# Model name → config path shorthand
# ---------------------------------------------------------------------------
MODEL_CONFIG_MAP = {
    'llama3.1-8b': 'llama3.1-8b.yaml',
    'llama2-7b': 'llama2-7b.yaml',
    'llama-7b': 'llama-7b.yaml',
    'mistral-7b': 'mistral-7b.yaml',
    'llama2-13b': 'llama-13b.yaml',
    'llama-13b': 'llama-13b.yaml',
    'qwen3-14b': 'qwen3-14b.yaml',
    'gpt-j-6B': 'gpt-j-6B.yaml',
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Run parameter-based knowledge editing experiments'
    )
    # Method & model
    p.add_argument('--method', required=True, choices=list(METHOD_HPARAMS.keys()),
                   help='Editing method')
    p.add_argument('--model', required=True,
                   help='Target LLM (e.g. llama3.1-8b, mistral-7b, llama2-13b)')
    p.add_argument('--config_dir', default='configs',
                   help='Path to configs/ directory (default: configs/)')

    # Dataset
    p.add_argument('--dataset', default='zsre',
                   choices=['zsre', 'wiki_counterfact', 'counterfact', 'elken'],
                   help='Dataset for editing')
    p.add_argument('--data_path', required=True,
                   help='Path to the dataset JSON file')
    p.add_argument('--train_data_path', default=None,
                   help='Path to training split (required for IKE)')

    # Editing scenario
    p.add_argument('--N', type=int, default=None,
                   help='Number of edits (default: all)')
    p.add_argument('--sequential', action='store_true',
                   help='Use sequential editing (default: single/batch)')
    p.add_argument('--start_index', type=int, default=None,
                   help='Start index in dataset')
    p.add_argument('--end_index', type=int, default=None,
                   help='End index in dataset')

    # Evaluation
    p.add_argument('--eval_mode', default='llm_judge',
                   choices=['exact_match', 'llm_judge'],
                   help='Evaluation metric type')
    p.add_argument('--api_key', default='dummy',
                   help='API key for LLM judge service')

    # Output
    p.add_argument('--output_dir', default='outputs',
                   help='Directory to save results')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')

    return p.parse_args()


def load_dataset(args):
    """Load and prepare the editing dataset."""
    print(f"Loading dataset from {args.data_path} ...")
    dataset = KnowEditDataset(
        args.data_path,
        size=args.N,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    prompts = [d['prompt'] for d in dataset]
    subjects = [d['subject'] for d in dataset]
    target_new = [d['target_new'] for d in dataset]
    rephrase_prompts = [d['rephrase_prompt'] for d in dataset]

    # --- Portability ---
    portability_inputs = {}
    for key, port_key in [('Subject_Aliasing', 'portability_s'),
                           ('reasoning', 'portability_r'),
                           ('Logical_Generalization', 'portability_l')]:
        prompts_list, answers_list = [], []
        for d in dataset:
            items = d.get(port_key, [])
            p_list, a_list = [], []
            if items:
                for item in items:
                    if item and item.get('prompt') and item.get('ground_truth'):
                        ans = item['ground_truth']
                        if isinstance(ans, list):
                            ans = ans[0]
                        if ans.strip():
                            p_list.append(item['prompt'])
                            a_list.append(ans)
            prompts_list.append(p_list if p_list else None)
            answers_list.append(a_list if a_list else None)
        portability_inputs[key] = {'prompt': prompts_list, 'ground_truth': answers_list}

    # --- Locality ---
    locality_inputs = {}
    for key, loc_key in [('Relation_Specificity', 'locality_rs'),
                          ('Forgetfulness', 'locality_f')]:
        prompts_list, answers_list = [], []
        for d in dataset:
            items = d.get(loc_key, [])
            p_list, a_list = [], []
            if items:
                for item in items:
                    if item and item.get('prompt') and item.get('ground_truth'):
                        ans = item['ground_truth']
                        if isinstance(ans, list):
                            ans = ans[0]
                        if ans.strip():
                            p_list.append(item['prompt'])
                            a_list.append(ans)
            prompts_list.append(p_list if p_list else None)
            answers_list.append(a_list if a_list else None)
        locality_inputs[key] = {'prompt': prompts_list, 'ground_truth': answers_list}

    return {
        'prompts': prompts,
        'subjects': subjects,
        'target_new': target_new,
        'rephrase_prompts': rephrase_prompts,
        'locality_inputs': locality_inputs,
        'portability_inputs': portability_inputs,
        'dataset': dataset,
    }


def get_hparams_path(method, model_short, config_dir):
    """Resolve config file path."""
    config_file = MODEL_CONFIG_MAP.get(model_short, f'{model_short}.yaml')
    path = os.path.join(config_dir, method, config_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config not found: {path}. "
            f"Check that configs/{method}/ contains the config for {model_short}."
        )
    return path


def build_editor(method, hparams_path, eval_mode, api_key):
    """Instantiate the editor with the correct hyperparameters."""
    hparams_cls = METHOD_HPARAMS[method]
    hparams = hparams_cls.from_hparams(hparams_path)
    hparams.evaluation_type = eval_mode
    hparams.api_key = api_key
    editor = BaseEditor.from_hparams(hparams)
    return editor, hparams


def main():
    args = parse_args()
    random.seed(args.seed)

    # --- Resolve config ---
    hparams_path = get_hparams_path(args.method, args.model, args.config_dir)
    print(f"Method: {args.method} | Model: {args.model}")
    print(f"Config: {hparams_path}")
    print(f"Dataset: {args.dataset} | N: {args.N or 'all'} | "
          f"Sequential: {args.sequential} | Eval: {args.eval_mode}")

    # --- Load data ---
    data = load_dataset(args)

    # --- Build editor ---
    editor, hparams = build_editor(args.method, hparams_path, args.eval_mode, args.api_key)

    # --- IKE pre-processing ---
    train_ds = None
    if args.method == 'IKE':
        if args.train_data_path is None:
            raise ValueError("--train_data_path is required for IKE")
        if args.dataset == 'zsre':
            train_ds = ZsreDataset(args.train_data_path)
        else:
            train_ds = WikiCounterfactDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(
            f'cuda:{hparams.device}'
        )
        encode_ike_facts(sentence_model, train_ds, hparams)

    # --- WISE locality setup ---
    extra_kwargs = {}
    if args.method == 'WISE':
        # WISE needs loc_prompts for its memory construction
        extra_kwargs['loc_prompts'] = [
            d.get('loc', '') + ' ' + d.get('loc_ans', '')
            for d in data['dataset']
        ]

    # --- Run editing ---
    metrics, edited_model, _ = editor.edit(
        prompts=data['prompts'],
        rephrase_prompts=data['rephrase_prompts'],
        target_new=data['target_new'],
        subject=data['subjects'],
        locality_inputs=data['locality_inputs'],
        portability_inputs=data['portability_inputs'],
        train_ds=train_ds,
        sequential_edit=args.sequential,
        test_generation=False,
        **extra_kwargs,
    )

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = (
        f'{args.dataset}_{args.method}_N={args.N or "all"}_'
        f'Sequential={args.sequential}_{timestamp}.json'
    )
    output_path = os.path.join(args.output_dir, fname)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # --- Summary ---
    if metrics:
        from easyeditor.evaluate.evaluate_utils import summary_metrics_for_LLM_judge
        summary = summary_metrics_for_LLM_judge(metrics)
        print("\n=== Results Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
