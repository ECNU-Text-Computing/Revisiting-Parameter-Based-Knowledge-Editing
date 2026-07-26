import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_representation(h_original, energy_threshold=0.99, save_dir=None, prefix="orig"):
    """
    对原始表示进行中心化、SVD分解，计算相关指标，并保存结果。

    参数:
        h_original: torch.Tensor, 形状 [n_samples, d]
        energy_threshold: float, 用于定义有效秩的能量阈值 (默认0.99)
        save_dir: str or Path, 结果保存目录。如果为None，则不保存到文件。
        prefix: str, 保存文件名的前缀，用于区分不同实验。

    返回:
        svd_result: dict, 包含所有基准数据。
    """
    # 1. 转置与中心化
    H = h_original.T.float()  # 形状 [d, n]
    mean_vec = H.mean(dim=1, keepdim=True)  # [d, 1]
    H_centered = H - mean_vec  # [d, n]

    # 2. 进行经济型SVD
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    # U: [d, n], S: [n], Vh: [n, n]

    # 3. 计算指标
    # 奇异值平方
    S_sq = S ** 2
    total_energy = S_sq.sum()
    cum_energy = S_sq.cumsum(dim=0)  # 累积能量
    explained_variance_ratio = cum_energy / total_energy  # 累积解释方差比例

    # 计算有效秩 r_eff: 第一个使累积解释方差 >= threshold 的索引 (1-based 转为 0-based)
    r_eff = (explained_variance_ratio >= energy_threshold).nonzero()[0].item() + 1
    sigma_min = S[r_eff - 1]  # 最小非零奇异值 (对应有效秩)
    sigma_max = S[0]  # 最小非零奇异值 (对应有效秩)

    # 计算数值秩 (矩阵的数值秩估计)
    numerical_rank = (total_energy ** 2) / (S_sq @ S_sq)  # (sum σ_i^2)^2 / sum σ_i^4
    numerical_rank = numerical_rank.item()

    # 4. 组装结果
    svd_result = {
        'U': U,                         # [d, n]
        'S': S,                         # [n]
        'mean_vec': mean_vec,           # [d, 1]
        'r_eff': r_eff,
        'sigma_min': sigma_min.item(),
        'energy_threshold': energy_threshold,
        'explained_variance_ratio': explained_variance_ratio,  # 累积解释方差比例向量
        'numerical_rank': numerical_rank,
        'total_energy': total_energy.item(),
        'n_samples': h_original.shape[0],
        'd': h_original.shape[1],
    }

    # 5. 可视化 (可选，但强烈推荐)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 5.1 奇异值衰减曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(S.numpy(), marker='.', linestyle='-', markersize=3)
        plt.axvline(x=r_eff-1, color='r', linestyle='--', label=f'r_eff={r_eff}')
        plt.yscale('log')
        plt.xlabel('Component index (k)')
        plt.ylabel('Singular value (σ_k) [log scale]')
        plt.title('Singular Value Spectrum')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # 5.2 累积解释方差曲线
        plt.subplot(1, 2, 2)
        plt.plot(explained_variance_ratio.numpy(), marker='.', linestyle='-', markersize=3)
        plt.axhline(y=energy_threshold, color='g', linestyle='--', label=f'{energy_threshold*100}% threshold')
        plt.axvline(x=r_eff-1, color='r', linestyle='--', label=f'r_eff={r_eff}')
        plt.xlabel('Number of components (k)')
        plt.ylabel('Cumulative explained variance ratio')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plot_path = save_dir / f"{prefix}_svd_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"可视化图表已保存至: {plot_path}")

        # 5.3 保存SVD结果 (元数据和关键张量)
        # 使用 torch.save 保存张量，用 json 保存元数据
        data_path = save_dir / f"{prefix}_svd_result.pt"
        # 保存所有张量和元数据
        torch.save(svd_result, data_path)
        print(f"SVD结果 (含张量) 已保存至: {data_path}")

        # 5.4 额外保存一份人类可读的摘要
        summary = {
            'r_eff': r_eff,
            'sigma_min': float(sigma_min),
            'sigma_max': float(sigma_max),
            'energy_threshold': energy_threshold,
            # 'numerical_rank': numerical_rank,
            'total_energy': float(total_energy),
            'n_samples': h_original.shape[0],
            'd': h_original.shape[1],
        }
        summary_path = save_dir / f"{prefix}_svd_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"SVD分析摘要已保存至: {summary_path}")

    return svd_result
if __name__ == "__main__":
    # 1. 加载你的原始数据
    import os
    import argparse

    # dataset = 'c4_en'
    # TARGET_LAYER_INDICES = [5,15,27,28,29,30,31]

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SVD analysis for different layers and datasets')
    parser.add_argument('--model', type=str, required=True, 
                       help='Dataset name (e.g., llama3.1-8b)')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (e.g., c4_en)')
    parser.add_argument('--layers', type=str, required=True,
                       help='Comma-separated list of layer indices (e.g., 5,15,27,28,29,30,31)')
    args = parser.parse_args()
    
    # 解析层索引参数
    TARGET_LAYER_INDICES = [int(idx) for idx in args.layers.split(',')]
    dataset = args.dataset
    model = args.model

    for idx in TARGET_LAYER_INDICES:
        H_PATH = f'/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/empirical_exp/exp_data/{model}/{dataset}/00_original/layer_{idx}/h_original.pt'
        save_dir = f"/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/empirical_exp/results/{model}/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        
        h_original = torch.load(H_PATH)  # [1301, 4096]
        # 2. 进行基准SVD分析
        svd_result = analyze_representation(
            h_original,
            energy_threshold=0.99,
            save_dir=save_dir,
            prefix=f"layer{idx}_edit0"
        )
