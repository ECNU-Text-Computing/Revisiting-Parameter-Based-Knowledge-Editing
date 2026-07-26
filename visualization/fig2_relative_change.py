#!/usr/bin/env python
"""
Figure 2: Distribution of Rk values (Directional Relative Change Rate).

Reproduces Figure 2 in the paper: shows how editing-induced perturbations are
disproportionately amplified in low-singular-value directions, confirming
Theorem 4.5 (Relative Amplification in Low-Scale Directions).

The script loads pre-computed Rk values from the theory/ data and generates
a scatter/distribution plot showing Rk vs. singular value rank.

Usage:
    python fig2_relative_change.py --data_dir ../theory/ --output fig2_rk_distribution.pdf
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Generate Figure 2: Rk distribution')
    p.add_argument('--data_dir', default='../theory',
                   help='Path to theory data directory')
    p.add_argument('--output', default='fig2_rk_distribution.pdf')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    # Try to load pre-computed Rk data
    data_path = os.path.join(args.data_dir, 'principal_angles_analysis_results.csv')
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows from {data_path}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run theory/relative_change_rate.py first to compute Rk values.")
        return

    # Plot: Rk vs singular value rank
    fig, ax = plt.subplots(figsize=(8, 5))

    if 'singular_value_rank' in df.columns and 'Rk' in df.columns:
        ax.scatter(df['singular_value_rank'], df['Rk'], alpha=0.3, s=2, c='steelblue')
        ax.set_xlabel('Singular Value Rank (descending)')
        ax.set_ylabel('$R_k$ (Directional Relative Change Rate)')
        ax.set_title('Figure 2: Distribution of $R_k$ values (Layer 30, MEMIT, 1000 edits)')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='$R_k=1$ threshold')
        ax.set_yscale('log')
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
