#!/usr/bin/env python
"""
Figure 3: Knowledge editing performance under practice-oriented settings (RQ1).

Plots Reliability, Generalization, Locality, and Portability as the number of
sequential edits increases (1 → 10 → 100 → full dataset), comparing multiple
parameter-based methods against the SCR retrieval baseline.

Usage:
    python fig3_editing_performance.py --results_dir ../outputs/ --output fig3_performance.pdf
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = ['ROME', 'MEMIT', 'PMET', 'AlphaEdit', 'FT', 'MEND', 'WISE', 'LoRA', 'GRACE', 'IKE', 'SCR']
DIMENSIONS = ['reliability', 'generalization', 'locality', 'portability']
EDIT_COUNTS = [1, 10, 100, 1000]


def parse_args():
    p = argparse.ArgumentParser(description='Generate Figure 3: editing performance')
    p.add_argument('--results_dir', default='../outputs')
    p.add_argument('--output', default='fig3_editing_performance.pdf')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def load_results(results_dir):
    """Load all JSON result files and organize by (method, N)."""
    data = {}
    for fpath in glob.glob(os.path.join(results_dir, '*.json')):
        fname = os.path.basename(fpath)
        # Expected format: {dataset}_{method}_N={N}_Sequential={seq}_{timestamp}.json
        parts = fname.split('_')
        try:
            method = parts[1] if len(parts) > 1 else 'unknown'
            with open(fpath) as f:
                metrics = json.load(f)
            # Try to parse N from filename
            for part in parts:
                if part.startswith('N='):
                    n_val = part.split('=')[1]
                    n_key = int(n_val) if n_val != 'None' else 1000
                    break
            else:
                n_key = 'unknown'
            data.setdefault(method, {})[n_key] = metrics
        except Exception:
            continue
    return data


def main():
    args = parse_args()
    data = load_results(args.results_dir)

    if not data:
        print(f"No result files found in {args.results_dir}")
        print("Run experiments/run_param_edit.py first to generate results.")
        return

    # Create subplot grid: 4 dimensions × 1 row, each showing N on x-axis
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for method_idx, method in enumerate(METHODS):
            if method not in data:
                continue
            xs, ys = [], []
            for n in sorted(data[method].keys()):
                try:
                    n_int = int(n)
                except (ValueError, TypeError):
                    continue
                # Extract metric
                metrics_list = data[method][n]
                vals = []
                for m in metrics_list:
                    post = m.get('post', {})
                    if dim in post:
                        vals.append(post[dim])
                if vals:
                    xs.append(n_int)
                    ys.append(np.mean(vals))
            if xs:
                ax.plot(xs, ys, 'o-', color=colors[method_idx], label=method, markersize=4)
        ax.set_title(dim.capitalize())
        ax.set_xlabel('Number of edits')
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)

    axes[0].legend(loc='lower left', fontsize=6, ncol=2)
    fig.suptitle('Figure 3: Editing performance vs. number of sequential edits (Llama-3.1-8B, ZsRE)')
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
