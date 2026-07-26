"""
奇异值谱分析模块
用于分析模型编辑前后的维度坍缩现象
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
from scipy.linalg import svd
import argparse
from pathlib import Path
import json
import torch



class SingularValueAnalyzer:
    """奇异值谱分析器"""

    def __init__(self, n_samples: int = 1024, threshold_ratio: float = 0.01):
        self.n_samples = n_samples
        self.threshold_ratio = threshold_ratio

    # =========================
    # SVD
    # =========================
    def compute_svd(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, sigma, Vt = svd(H, full_matrices=False)
        return U, sigma, Vt

    # =========================
    # Effective rank (entropy)
    # =========================
    def compute_effective_rank(self, sigma: np.ndarray, eps=1e-12) -> float:
        sigma = np.array(sigma)
        sigma = sigma[sigma > eps]

        if len(sigma) == 0:
            return 0.0

        p = sigma / (np.sum(sigma) + eps)
        H = -np.sum(p * np.log(p + eps))
        return float(np.exp(H))

    # =========================
    # PCA energy rank
    # =========================
    def num_principal_components(self, sigma: np.ndarray, threshold=0.9) -> int:
        sigma = np.array(sigma)
        energy = sigma ** 2

        if len(energy) == 0:
            return 0

        cum_energy = np.cumsum(energy) / (np.sum(energy) + 1e-12)
        k = np.searchsorted(cum_energy, threshold) + 1
        return int(k)

    # =========================
    # spectral decay exponent
    # =========================
    def estimate_alpha(self, sigma: np.ndarray, k: int = 50) -> float:
        sigma = np.array(sigma)
        sigma = np.maximum(sigma, 1e-12)

        i = np.arange(1, min(k, len(sigma)) + 1)
        log_i = np.log(i)
        log_sigma = np.log(sigma[:len(i)])

        coeffs = np.polyfit(log_i, log_sigma, 1)
        return float(-coeffs[0])

    # =========================
    # main analysis
    # =========================
    def analyze_spectrum(
        self,
        H_original: np.ndarray,
        H_edited: np.ndarray,
        layer_name: str = "layer"
    ) -> Dict:

        U_original, sigma_original, _ = self.compute_svd(H_original)
        U_edited, sigma_edited, _ = self.compute_svd(H_edited)

        d,n = H_original.shape

        # ===== effective rank =====
        r_original = self.compute_effective_rank(sigma_original)
        r_edited = self.compute_effective_rank(sigma_edited)

        # ===== PCA components =====
        k_original = self.num_principal_components(sigma_original, 0.9)
        k_edited = self.num_principal_components(sigma_edited, 0.9)

        # ===== spectral decay =====
        alpha_original = self.estimate_alpha(sigma_original)
        alpha_edited = self.estimate_alpha(sigma_edited)

        # ===== min singular value =====
        sigma_min_original = float(np.min(sigma_original)) if len(sigma_original) > 0 else 0.0
        sigma_min_edited = float(np.min(sigma_edited)) if len(sigma_edited) > 0 else 0.0

        # ===== condition number =====
        cond_original = float(sigma_original[0] / (sigma_min_original + 1e-12))
        cond_edited = float(sigma_edited[0] / (sigma_min_edited + 1e-12))

        # ===== tail ratio =====
        tail_idx = min(50, len(sigma_original) - 1) if len(sigma_original) > 1 else 0
        tail_ratio_original = float(sigma_original[tail_idx] / (sigma_original[0] + 1e-12))
        tail_ratio_edited = float(sigma_edited[min(50, len(sigma_edited)-1)] / (sigma_edited[0] + 1e-12))

        # ===== collapse ratio =====
        collapse_original = np.mean(
            sigma_original < sigma_original[0] * self.threshold_ratio
        ) if len(sigma_original) > 0 else 0.0

        collapse_edited = np.mean(
            sigma_edited < sigma_edited[0] * self.threshold_ratio
        ) if len(sigma_edited) > 0 else 0.0

        return {
            "layer_name": layer_name,
            "dimension": d,
            "n_samples": n,

            # spectra
            "sigma_original": sigma_original.tolist(),
            "sigma_edited": sigma_edited.tolist(),

            # rank
            "effective_rank_original": r_original,
            "effective_rank_edited": r_edited,
            "rank_change": r_edited - r_original,

            # PCA
            "principal_components_original": k_original,
            "principal_components_edited": k_edited,

            # sigma_min_original
            "sigma_min_original": sigma_min_original,
            "sigma_min_edited": sigma_min_edited,

            # decay
            "alpha_original": alpha_original,
            "alpha_edited": alpha_edited,

            # conditioning
            "cond_original": cond_original,
            "cond_edited": cond_edited,

            # tails
            "tail_ratio_original": tail_ratio_original,
            "tail_ratio_edited": tail_ratio_edited,

            # collapse
            "collapse_ratio_original": float(collapse_original),
            "collapse_ratio_edited": float(collapse_edited),

            # U
            "U_original": U_original,
            "U_edited": U_edited,
        }

    # =========================
    # visualization
    # =========================
    def visualize_spectrum_comparison(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        figsize=(16, 10)
    ):

        sigma_original = np.array(results["sigma_original"])
        sigma_edited = np.array(results["sigma_edited"])

        r_original = results["effective_rank_original"]
        r_edited = results["effective_rank_edited"]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # =========================
        # 1 spectrum
        # =========================
        ax1 = fig.add_subplot(gs[0, 0])
        k = np.arange(1, len(sigma_original) + 1)

        ax1.semilogy(k, sigma_original, label="Original", linewidth=2)
        ax1.semilogy(k, sigma_edited, label="Edited", linewidth=2)

        ax1.axvline(r_original, linestyle=":", alpha=0.5)
        ax1.axvline(r_edited, linestyle=":", alpha=0.5)

        ax1.set_title("Singular Value Spectrum")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # =========================
        # 2 change rate
        # =========================
        ax2 = fig.add_subplot(gs[0, 1])

        min_len = min(len(sigma_original), len(sigma_edited))
        if min_len > 0:
            change = (
                sigma_edited[:min_len] - sigma_original[:min_len]
            ) / (sigma_original[:min_len] + 1e-12)

            ax2.plot(change * 100)
            ax2.axhline(0, color="black", alpha=0.3)
            ax2.set_title("Change Rate (%)")
            ax2.grid(True, alpha=0.3)

        # =========================
        # 3 energy
        # =========================
        ax3 = fig.add_subplot(gs[0, 2])

        e1 = np.cumsum(sigma_original**2) / (np.sum(sigma_original**2) + 1e-12)
        e2 = np.cumsum(sigma_edited**2) / (np.sum(sigma_edited**2) + 1e-12)

        ax3.plot(e1, label="Original")
        ax3.plot(e2, label="Edited")
        ax3.axhline(0.9, linestyle=":")
        ax3.legend()
        ax3.set_title("Cumulative Energy")

        # =========================
        # 4 collapse
        # =========================
        ax4 = fig.add_subplot(gs[1, 0])

        ax4.bar(
            ["Orig", "Edit"],
            [
                results["collapse_ratio_original"],
                results["collapse_ratio_edited"],
            ],
        )
        ax4.set_title("Collapse Ratio")

        # =========================
        # 5 rank
        # =========================
        ax5 = fig.add_subplot(gs[1, 1])

        ax5.bar(
            ["Orig", "Edit"],
            [r_original, r_edited],
        )
        ax5.set_title("Effective Rank")

        # =========================
        # 6 min sigma
        # =========================
        ax6 = fig.add_subplot(gs[1, 2])

        ax6.bar(
            ["Orig", "Edit"],
            [np.min(sigma_original), np.min(sigma_edited)],
        )
        ax6.set_yscale("log")
        ax6.set_title("Min Singular Value")

        fig.suptitle(f"{results['layer_name']} Spectrum Analysis", fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()




def load_hidden_state(path: str, module: str):

    x = torch.load(path, map_location="cpu")

    if isinstance(x, dict):

        if module in x:
            x = x[module]
        else:
            candidates = [k for k in x.keys() if module in k]

            if len(candidates) == 0:
                raise KeyError(f"module={module} not found, keys={list(x.keys())}")

            x = x[candidates[0]]

    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.float().cpu().numpy()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.shape}")

    return x


def analyze_one_layer(analyzer, orig_path, edit_path, layer_name, module):

    H_orig = load_hidden_state(orig_path, module)
    H_edit = load_hidden_state(edit_path, module)

    # shape = (N, D)
    if H_orig.shape[0] < H_orig.shape[1]:
        H_orig = H_orig
        H_edit = H_edit
    else:
        H_orig = H_orig.T
        H_edit = H_edit.T

    # center
    mean_vec = H_orig.mean(axis=0, keepdims=True)

    H_o = H_orig - mean_vec
    H_e = H_edit - mean_vec

    return analyzer.analyze_spectrum(H_o, H_e, layer_name)



def to_json_safe(obj):
    """递归转 JSON-safe"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


def split_result(result):
    """
    split into:
    - json-safe metrics
    - matrix (np / torch)
    """
    json_result = {}
    matrix_result = {}

    for k, v in result.items():

        # torch tensor
        if hasattr(v, "detach"):
            matrix_result[k] = v.detach().cpu().numpy()

        # numpy array
        elif isinstance(v, np.ndarray):
            matrix_result[k] = v

        # scalar / list / dict
        else:
            json_result[k] = v

    return json_result, matrix_result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--orig_model", type=str, required=True)
    parser.add_argument("--edit_steps", type=str, nargs="+", required=True)

    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--module", type=str, default="down_proj_last")

    parser.add_argument("--save_dir", type=str, default="./svd_results")

    args = parser.parse_args()

    analyzer = SingularValueAnalyzer()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_all = {}

    orig_base = Path(args.base_dir) / args.orig_model / args.dataset / args.module
    print(f"\n📌 Original model: {orig_base}")

    for step in args.edit_steps:

        print("\n====================================")
        print(f"🚀 Processing edit step: {step}")

        edit_base = Path(args.base_dir) / step / args.dataset / args.module

        step_summary = []

        for L in args.layers:

            orig_path = orig_base / f"layer_{L}.pt"
            edit_path = edit_base / f"layer_{L}.pt"

            if not orig_path.exists() or not edit_path.exists():
                print(orig_path)
                print(edit_path)
                print(f"⚠️ skip layer {L}, missing file")
                continue

            print(f"  🔹 Layer {L}")

            parts = args.module.split("_")
            module_base = "_".join(parts[:-1]) if len(parts) > 1 else args.module
            module = f"layer{L}_{module_base}"

            # =========================
            # run analysis
            # =========================
            result = analyze_one_layer(
                analyzer,
                str(orig_path),
                str(edit_path),
                layer_name=f"layer_{L}_step_{step}",
                module=module
            )

            # =========================
            # split result
            # =========================
            json_result, matrix_result = split_result(result)

            # =========================
            # save JSON (metrics only)
            # =========================
            layer_json_path = save_dir / f"{step}_layer_{L}.json"
            with open(layer_json_path, "w") as f:
                json.dump(to_json_safe(json_result), f, indent=2)

            # =========================
            # save matrices (U, sigma, etc.)
            # =========================
            if len(matrix_result) > 0:
                layer_npz_path = save_dir / f"{step}_layer_{L}_matrices.npz"
                np.savez_compressed(layer_npz_path, **matrix_result)

            # =========================
            # visualization
            # =========================
            fig_path = save_dir / f"{step}_layer_{L}.png"
            analyzer.visualize_spectrum_comparison(
                result,
                save_path=str(fig_path)
            )

            # =========================
            # store summary (small only)
            # =========================
            step_summary.append({
                "layer": L,
                "json_file": str(layer_json_path),
                "matrix_file": str(save_dir / f"{step}_layer_{L}_matrices.npz") if len(matrix_result) > 0 else None,
            })

        results_all[step] = step_summary

    # =========================
    # global summary (safe)
    # =========================
    summary_path = save_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print("\n🎉 All analysis completed!")
    print(f"📁 Saved to: {save_dir}")


if __name__ == "__main__":
    main()