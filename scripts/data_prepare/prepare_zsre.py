#!/usr/bin/env python
"""
Prepare ZsRE dataset for knowledge editing experiments.

The ZsRE dataset (Levy et al., 2017) is a zero-shot relation extraction benchmark
commonly used for knowledge editing evaluation. Each entry contains a subject,
relation, prompt, rephrase, locality queries, and portability queries.

Source: https://huggingface.co/datasets/zjunlp/KnowEdit

Usage:
    python prepare_zsre.py --output_dir ./data/zsre
"""

import os
import json
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser(description='Prepare ZsRE dataset')
    p.add_argument('--output_dir', default='./data/zsre', help='Output directory')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("ZsRE dataset loading instructions:")
    print("  1. Visit https://huggingface.co/datasets/zjunlp/KnowEdit")
    print("  2. Download the ZsRE subset files:")
    print("     - zsre_mend_eval.json  (evaluation)")
    print("     - zsre_mend_train.json (training, for IKE)")
    print(f"  3. Place them in: {os.path.abspath(args.output_dir)}")
    print()
    print("Dataset format per entry:")
    print("  {")
    print('    "src": "What university did Watts Humphrey attend?",')
    print('    "answers": ["Illinois Institute of Technology"],')
    print('    "alt": "University of Michigan",')
    print('    "subject": "Watts Humphrey",')
    print('    "rephrase": "Which university was Watts Humphrey educated at?",')
    print('    "loc": "Watts Humphrey was a...",')
    print('    "loc_ans": "software engineer",')
    print('    "portability": {')
    print('      "New Question": "...",')
    print('      "New Answer": "..."')
    print('    }')
    print("  }")


if __name__ == '__main__':
    main()
