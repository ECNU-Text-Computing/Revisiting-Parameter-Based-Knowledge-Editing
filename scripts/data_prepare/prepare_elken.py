#!/usr/bin/env python
"""
Prepare ELKEN (Event-Level Knowledge Editing) dataset.

ELKEN (Peng et al., 2024) tests knowledge editing on complex event-level knowledge
rather than simple factual triples. Each event involves multiple entities and
attributes occurring concurrently.

The paper uses GPT-4o to transform raw event data into editing-compatible format.
See Appendix C.2 for full details.

Source: https://github.com/zjunlp/KnowEdit

Usage:
    python prepare_elken.py --input_dir /path/to/raw/elken --output_dir ./data/elken
"""

import os
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Prepare ELKEN event knowledge dataset')
    p.add_argument('--input_dir', help='Path to raw ELKEN data')
    p.add_argument('--output_dir', default='./data/elken')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("ELKEN dataset preparation:")
    print("  1. Download from: https://github.com/zjunlp/KnowEdit")
    print("     (ELKEN subset in the KnowEdit benchmark)")
    print()
    print("  2. Process with GPT-4o (see Appendix C.2):")
    print("     - Transform raw event descriptions into prompt-answer pairs")
    print("     - Each event → m prompt-answer pairs (x_i, y_i)")
    print("     - Generate rephrase, locality, and portability variants")
    print()
    print("  3. Output format per event:")
    print("  {")
    print('    "event_id": "E001",')
    print('    "prompt": "In 2023, Company X acquired Company Y for...",')
    print('    "target_new": "$5 billion",')
    print('    "subject": "Company X acquisition",')
    print('    "rephrase_prompt": "Company Y was bought by Company X in 2023 for...",')
    print('    "locality_rs": [...],')
    print('    "portability_s": [...]')
    print("  }")
    print()
    print(f"  Place processed files in: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
