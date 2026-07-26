#!/usr/bin/env python
"""
Figure 5: Trade-off between editing time and inference latency (RQ4).

Scatter plot showing editing time (x-axis) vs. normalized inference latency
(y-axis) for all evaluated methods. Reveals the fundamental stability-efficiency
trade-off discussed in the paper.

Usage:
    python fig5_efficiency.py --results_dir ../outputs/ --output fig5_efficiency.pdf
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PARAM_METHODS = ['ROME', 'MEMIT', 'PMET', 'AlphaEdit', 'FT', 'MEND', 'WISE', 'LoRA']
EXT_MEM_METHODS = ['GRACE', 'IKE', 'SCR']


def parse_args():
    p = argparse.ArgumentParser(description='Generate Figure 5: efficiency trade-off')
    p.add_argument('--results_dir', default='../outputs')
    p.add_argument('--output', default='fig5_efficiency.pdf')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    efficiency_files = glob.glob(os.path.join(args.results_dir, 'efficiency_*.json'))
    if not efficiency_files:
        print(f"No efficiency files found in {args.results_dir}")
        print("Run experiments/run_efficiency.py first to generate results.")
        return

    methods, edit_times, norm_latencies, colors = [], [], [], []
    for fpath in efficiency_files:
        with open(fpath) as f:
            data = json.load(f)
        method = data.get('method', 'unknown')
        methods.append(method)
        edit_times.append(data.get('avg_edit_time_s', 0))
        norm_latencies.append(data.get('normalized_latency', 1.0))
        if method in PARAM_METHODS:
            colors.append('steelblue')
        else:
            colors.append('darkorange')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(edit_times, norm_latencies, c=colors, s=80, alpha=0.8, edgecolors='black')

    for i, method in enumerate(methods):
        ax.annotate(method, (edit_times[i], norm_latencies[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Average Edit Time (seconds)')
    ax.set_ylabel('Normalized Inference Latency (× base model)')
    ax.set_title('Figure 5: Editing Time vs. Inference Latency Trade-off')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Base model latency')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Parameter-based methods'),
        Patch(facecolor='darkorange', label='External memory methods'),
    ]
    ax.legend(handles=legend_elements)

    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
