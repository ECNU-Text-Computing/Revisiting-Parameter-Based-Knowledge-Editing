#!/usr/bin/env python
"""
Figure 4: Event-level knowledge editing performance (RQ3).

Compares AlphaEdit and SCR on sequential editing of complex event knowledge
from the ELKEN dataset. Shows that parameter-based methods struggle with
multi-entity, multi-attribute event knowledge.

Usage:
    python fig4_event_editing.py --results_dir ../outputs/ --output fig4_event.pdf
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Generate Figure 4: event knowledge editing')
    p.add_argument('--results_dir', default='../outputs')
    p.add_argument('--output', default='fig4_event_editing.pdf')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    # Look for event result files
    event_files = glob.glob(os.path.join(args.results_dir, 'event_*.json'))
    if not event_files:
        print(f"No event result files found in {args.results_dir}")
        print("Run experiments/run_event_edit.py first to generate results.")
        return

    # Plot: Reliability and Generalization for AlphaEdit vs SCR on event data
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, metric in enumerate(['reliability', 'generalization']):
        ax = axes[ax_idx]
        for fpath in event_files:
            fname = os.path.basename(fpath)
            with open(fpath) as f:
                data = json.load(f)
            method = fname.split('_')[1] if '_' in fname else 'unknown'
            # Accumulate metric values
            vals = []
            for m in data:
                post = m.get('post', {})
                if metric in post:
                    vals.append(float(post[metric]))
            if vals:
                ax.bar(method, np.mean(vals), alpha=0.7)
        ax.set_title(f'Event Knowledge — {metric.capitalize()}')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')

    fig.suptitle('Figure 4: Event-level sequential editing performance (ELKEN)')
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
