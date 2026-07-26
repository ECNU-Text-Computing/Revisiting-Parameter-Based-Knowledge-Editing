#!/usr/bin/env python
"""
Run knowledge editing on event-level knowledge (RQ3).

Evaluates whether parameter-based editing methods can generalize from factual
triples to complex event knowledge (ELKEN dataset), where each edit involves
multiple entities and attributes.

Usage:
    python run_event_edit.py --method AlphaEdit --model llama3.1-8b \\
        --data_path /path/to/elken.json --N 100 --sequential
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
        description='Event-level knowledge editing experiments (RQ3)'
    )
    p.add_argument('--method', required=True, choices=list(METHOD_HPARAMS.keys()))
    p.add_argument('--model', required=True, help='Target LLM path or name')
    p.add_argument('--config_path', required=True, help='Path to hyperparameter YAML')
    p.add_argument('--data_path', required=True, help='Path to ELKEN dataset')
    p.add_argument('--N', type=int, default=None, help='Number of event edits')
    p.add_argument('--sequential', action='store_true', default=True)
    p.add_argument('--eval_mode', default='llm_judge',
                   choices=['exact_match', 'llm_judge'])
    p.add_argument('--output_dir', default='outputs')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"RQ3: Event editing {args.method} on {args.model} | "
          f"N={args.N or 'all'} | Sequential={args.sequential}")

    # Build editor
    hparams_cls = METHOD_HPARAMS[args.method]
    hparams = hparams_cls.from_hparams(args.config_path)
    hparams.evaluation_type = args.eval_mode
    editor = BaseEditor.from_hparams(hparams)

    # Load event dataset (ELKEN format: multi prompt-answer pairs per event)
    dataset = KnowEditDataset(args.data_path, size=args.N)
    prompts = [d['prompt'] for d in dataset]
    subjects = [d['subject'] for d in dataset]
    target_new = [d['target_new'] for d in dataset]
    rephrase_prompts = [d.get('rephrase_prompt', d['prompt']) for d in dataset]

    # Build portability and locality from event data
    locality_inputs = {}
    portability_inputs = {}

    for key, loc_key in [('Relation_Specificity', 'locality_rs')]:
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
        if any(p is not None for p in prompts_list):
            locality_inputs[key] = {'prompt': prompts_list, 'ground_truth': answers_list}

    # Run editing
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs if locality_inputs else None,
        portability_inputs=portability_inputs if portability_inputs else None,
        sequential_edit=args.sequential,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = f'event_{args.method}_{args.model}_N{args.N or "all"}_{timestamp}.json'
    with open(os.path.join(args.output_dir, fname), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Results saved to: {os.path.join(args.output_dir, fname)}")


if __name__ == '__main__':
    main()
