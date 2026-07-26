#!/usr/bin/env python
"""
Prepare WikiData-counterfact dataset for knowledge editing experiments.

WikiData-counterfact (Cohen et al., 2024) tests whether editing methods can
update factual triples without affecting related knowledge.

Source: https://huggingface.co/datasets/zjunlp/KnowEdit (counterfact split)

Usage:
    python prepare_wiki_counterfact.py --output_dir ./data/wiki_counterfact
"""

import os
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Prepare WikiData-counterfact dataset')
    p.add_argument('--output_dir', default='./data/wiki_counterfact')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("WikiData-counterfact dataset loading instructions:")
    print("  1. Visit https://huggingface.co/datasets/zjunlp/KnowEdit")
    print("  2. Download the WikiData-counterfact files:")
    print("     - test_cf.json     (evaluation)")
    print("     - train_cf.json    (for IKE training)")
    print(f"  3. Place them in: {os.path.abspath(args.output_dir)}")
    print()
    print("Dataset format per entry:")
    print("  {")
    print('    "prompt": "The official language of France is",')
    print('    "target_new": "German",')
    print('    "ground_truth": "French",')
    print('    "subject": "France",')
    print('    "rephrase_prompt": "In France, they speak",')
    print('    "locality": {...},')
    print('    "portability": {...}')
    print("  }")


if __name__ == '__main__':
    main()
