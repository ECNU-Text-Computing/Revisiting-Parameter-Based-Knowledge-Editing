import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union
import warnings


class ActivationAnalyzer:
    """激活值分析器，用于分析模型编辑前后的变化"""
    
    def __init__(self, 
                 data_dir: str = None,
                 output_dir: str = "./activation_analysis"):
        """
        初始化分析器
        
        Args:
            data_dir: 数据文件所在目录
            output_dir: 输出目录，用于保存图片和结果文件
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 数据存储
        self.sigma_original = None
        self.U_o = None
        self.U_e = None
        self.h_edit = None
        self.h_orig = None
        
        # 计算中间结果
        self.delta_h = None
        self.delta_h_norms = None
        self.C = None
        self.R_values = None
        self.k_index = None
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_json_data(self, json_path: Optional[str] = None) -> torch.Tensor:
        """
        从JSON文件加载sigma_original值
        
        Args:
            json_path: JSON文件路径，如果为None则使用默认路径
            
        Returns:
            sigma_original张量
        """
        if json_path is None and self.data_dir is not None:
            json_path = os.path.join(self.data_dir, 'Llama-3.1-8B-Instruct_AlphaEdit_num1_layer_30.json')
        elif json_path is None:
            raise ValueError("请提供JSON文件路径或设置data_dir")
        
        print(f"正在加载JSON文件: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if 'sigma_original' in data:
            sigma_original = np.array(data['sigma_original'])
            self.sigma_original = torch.from_numpy(sigma_original).float()
            print(f"成功提取到 sigma_original: shape={self.sigma_original.shape}")
        else:
            warnings.warn("JSON文件中未找到sigma_original键")
            self.sigma_original = None
            
        return self.sigma_original
    
    def load_npz_data(self, npz_path: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        从NPZ文件加载U矩阵
        
        Args:
            npz_path: NPZ文件路径，如果为None则使用默认路径
            verbose: 是否打印详细信息
            
        Returns:
            包含所有数组的字典
        """
        if npz_path is None and self.data_dir is not None:
            npz_path = os.path.join(self.data_dir, 'Llama-3.1-8B-Instruct_AlphaEdit_num1_layer_30_matrices.npz')
        elif npz_path is None:
            raise ValueError("请提供NPZ文件路径或设置data_dir")
        
        print(f"正在加载NPZ文件: {npz_path}")
        data = np.load(npz_path)
        
        if verbose:
            print("\n文件中的数组键名：")
            print(data.files)
            print()
            
            for key in data.files:
                array = data[key]
                print(f"数组名称: {key}")
                print(f"  形状: {array.shape}")
                print(f"  数据类型: {array.dtype}")
                print(f"  大小: {array.size}")
                if array.size > 5:
                    print(f"  前5个值示例: {array.flat[:5]}")
                else:
                    print(f"  值: {array.flatten()}")
                print("-" * 50)
        
        # 加载主要矩阵
        if 'U_original' in data:
            self.U_o = torch.from_numpy(data['U_original']).float()
        if 'U_edited' in data:
            self.U_e = torch.from_numpy(data['U_edited']).float()
            
        print(f"加载完成: U_original.shape={self.U_o.shape if self.U_o is not None else 'N/A'}, "
              f"U_edited.shape={self.U_e.shape if self.U_e is not None else 'N/A'}")
        
        return {key: data[key] for key in data.files}
    
    def load_pt_data(self, 
                     edit_path: Optional[str] = None, 
                     orig_path: Optional[str] = None) -> None:
        """
        加载.pt文件中的激活值
        
        Args:
            edit_path: 编辑后文件路径
            orig_path: 原始文件路径
        """
        if edit_path is None and self.data_dir is not None:
            edit_path = os.path.join(self.data_dir, 'layer_30.pt')
        if orig_path is None and self.data_dir is not None:
            orig_path = os.path.join(self.data_dir, 'ori_layer_30.pt')
        
        if edit_path is None or orig_path is None:
            raise ValueError("请提供.pt文件路径或设置data_dir")
        
        print(f"正在加载.pt文件:")
        print(f"  编辑后: {edit_path}")
        print(f"  原始: {orig_path}")
        
        # 加载到CPU避免显存问题
        h_edit = torch.load(edit_path, map_location='cpu')
        h_orig = torch.load(orig_path, map_location='cpu')
        
        # 提取layer30_down_proj
        if 'layer30_down_proj' in h_edit:
            self.h_edit = h_edit['layer30_down_proj'].float()
        elif isinstance(h_edit, torch.Tensor):
            self.h_edit = h_edit.float()
        else:
            raise KeyError("编辑文件中未找到layer30_down_proj键")
            
        if 'layer30_down_proj' in h_orig:
            self.h_orig = h_orig['layer30_down_proj'].float()
        elif isinstance(h_orig, torch.Tensor):
            self.h_orig = h_orig.float()
        else:
            raise KeyError("原始文件中未找到layer30_down_proj键")
        
        print(f"激活值加载完成: h_edit.shape={self.h_edit.shape}, h_orig.shape={self.h_orig.shape}")
    
    def load_all_data(self, 
                      json_path: Optional[str] = None,
                      npz_path: Optional[str] = None,
                      edit_path: Optional[str] = None,
                      orig_path: Optional[str] = None) -> None:
        """
        加载所有数据
        
        Args:
            json_path, npz_path, edit_path, orig_path: 各文件路径
        """
        print("="*60)
        print("开始加载所有数据")
        print("="*60)
        
        self.load_json_data(json_path)
        self.load_npz_data(npz_path)
        self.load_pt_data(edit_path, orig_path)
        
        print("\n" + "="*60)
        print("所有数据加载完成")
        print("="*60)
    
    def compute_perturbation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算扰动和投影
        
        Returns:
            (delta_h, C) 元组
        """
        if self.h_orig is None or self.h_edit is None or self.U_o is None:
            raise ValueError("请先加载所有必要数据")
        
        # 将数据移至CPU
        h_o_cpu = self.h_orig.cpu()
        h_e_cpu = self.h_edit.cpu()
        U_o_cpu = self.U_o.cpu()
        
        # 计算扰动
        self.delta_h = h_e_cpu - h_o_cpu  # [n_samples, d]
        n_samples, d = h_o_cpu.shape

        # 计算每个样本的L2范数
        self.delta_h_norms = torch.norm(self.delta_h, p=2, dim=1) 
        
        # 计算投影
        # 注意: U_o 的列是主成分，我们需要用 U_o^T 去左乘 delta_h^T
        self.C = torch.matmul(U_o_cpu.T, self.delta_h.T)  # [d, n_samples]
        
        print(f"扰动计算完成:")
        print(f"  delta_h.shape: {self.delta_h.shape}")
        print(f"  C.shape: {self.C.shape}")
        
        return self.delta_h, self.C, self.delta_h_norms
    
    def compute_R_values(self, epsilon: float = 1e-12) -> np.ndarray:
        """
        计算R_k值
        
        Args:
            epsilon: 极小奇异值的阈值
            
        Returns:
            R值数组
        """
        if self.C is None or self.sigma_original is None:
            self.compute_perturbation()
        
        n_samples, d = self.h_orig.shape
        sqrt_n = np.sqrt(n_samples)
        
        R_list = []
        sigma_orig_cpu = self.sigma_original.cpu()
        
        for k in range(d):
            if k >= self.C.shape[0]:
                warnings.warn(f"k={k} 超出C的维度{self.C.shape[0]}，跳过计算")
                break
                
            mean_abs_c_k = torch.mean(torch.abs(self.C[k, :]))
            sigma_k = sigma_orig_cpu[k]
            
            if sigma_k < epsilon:  # 处理极小奇异值
                R_k = 1e20
            else:
                R_k = (sqrt_n * mean_abs_c_k) / sigma_k
                
            R_list.append(R_k.item())
        
        self.R_values = np.array(R_list)
        self.k_index = np.arange(1, len(self.R_values) + 1)  # 主成分索引，从1开始
        
        print(f"R值计算完成: 共{len(self.R_values)}个值")
        print(f"  R_min: {np.min(self.R_values):.6f}")
        print(f"  R_max: {np.max(self.R_values):.6f}")
        print(f"  R_mean: {np.mean(self.R_values):.6f}")
        
        return self.R_values
    
    def analyze_R_statistics(self) -> Dict:
        """
        分析R值的统计信息
        
        Returns:
            统计信息字典
        """
        if self.R_values is None:
            self.compute_R_values()
        
        stats = {
            'mean': float(np.mean(self.R_values)),
            'median': float(np.median(self.R_values)),
            'min': float(np.min(self.R_values)),
            'max': float(np.max(self.R_values)),
            'std': float(np.std(self.R_values)),
            'num_values': len(self.R_values),
            'threshold_1': float(np.sum(self.R_values > 1.0)),
            'threshold_1_percent': float(np.mean(self.R_values > 1.0) * 100),
            'threshold_10': float(np.sum(self.R_values > 10.0)),
            'threshold_10_percent': float(np.mean(self.R_values > 10.0) * 100),
        }
        
        print("\n" + "="*60)
        print("R值统计分析")
        print("="*60)
        print(f"平均值: {stats['mean']:.4f}")
        print(f"中位数: {stats['median']:.4f}")
        print(f"最小值: {stats['min']:.4f}")
        print(f"最大值: {stats['max']:.4f}")
        print(f"标准差: {stats['std']:.4f}")
        print(f"R > 1.0: {stats['threshold_1']} 个 ({stats['threshold_1_percent']:.2f}%)")
        print(f"R > 10.0: {stats['threshold_10']} 个 ({stats['threshold_10_percent']:.2f}%)")
        
        return stats
    
    def save_results(self, 
                    stats: Dict = None,
                    filename: str = "activation_analysis_results.json") -> str:
        """
        保存分析结果到JSON文件
        
        Args:
            stats: 统计信息字典
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if stats is None:
            stats = self.analyze_R_statistics()
        
        # 添加额外的元信息
        results = {
            'analysis_time': np.datetime64('now').astype(str),
            'data_info': {
                'U_original_shape': list(self.U_o.shape) if self.U_o is not None else None,
                'U_edited_shape': list(self.U_e.shape) if self.U_e is not None else None,
                'h_original_shape': list(self.h_orig.shape) if self.h_orig is not None else None,
                'h_edited_shape': list(self.h_edit.shape) if self.h_edit is not None else None,
                'sigma_original_shape': list(self.sigma_original.shape) if self.sigma_original is not None else None,
            },
            'delta_h_norms_stats' : {
                'mean': float(self.delta_h_norms.mean().item()),
                'std': float(self.delta_h_norms.std().item()),
                'min': float(self.delta_h_norms.min().item()),
                'max': float(self.delta_h_norms.max().item()),
            },
            'R_statistics': stats,
            'delta_h_norms':self.delta_h_norms.tolist(),
            'R_values': self.R_values.tolist(),

        }
        
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\n分析结果已保存到: {file_path}")
        return file_path
    
    def save_R_values(self, filename: str = "R_values.npz") -> str:
        """
        保存R值和相关数据到.npz文件
        
        Args:
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if self.R_values is None:
            self.compute_R_values()
        
        file_path = os.path.join(self.output, filename)
        
        save_data = {
            'R_values': self.R_values,
            'k_index': self.k_index,
            'sigma_original': self.sigma_original.numpy() if self.sigma_original is not None else None,
        }
        
        np.savez(file_path, **save_data)
        print(f"R值数据已保存到: {file_path}")
        return file_path
    
    def plot_analysis(self, 
                     save_fig: bool = True,
                     fig_name: str = "activation_analysis.png",
                     dpi: int = 300,
                     figsize: Tuple[int, int] = (16, 6)) -> Optional[plt.Figure]:
        """
        绘制分析图表
        
        Args:
            save_fig: 是否保存图片
            fig_name: 图片文件名
            dpi: 图片分辨率
            figsize: 图片尺寸
            
        Returns:
            matplotlib图形对象，如果save_fig为False
        """
        if self.R_values is None:
            self.compute_R_values()
        
        sigma_np = self.sigma_original.numpy() if self.sigma_original is not None else None
        if sigma_np is None:
            raise ValueError("sigma_original未加载")
        
        # 确保维度匹配
        d = len(self.R_values)
        k_index = self.k_index[:d]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ---- 左子图: R_k 曲线 ----
        ax1.plot(k_index, self.R_values, 'b-', linewidth=2, label='Empirical R_k')
        ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='R=1 (Signal Level)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Principal Component Index (k)', fontsize=12)
        ax1.set_ylabel('Empirical Relative Change Rate (R_k)', fontsize=12)
        ax1.set_title('R_k vs. PC Index (Focus on Tail)', fontsize=14, fontweight='bold')
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend(fontsize=10)
        
        # 标记关键点
        if d > 0:
            ax1.axvline(x=d, color='gray', linestyle=':', alpha=0.7, label=f'k = d = {d}')
            ax1.legend(fontsize=10)
        
        # 添加统计信息
        stats_text = f"""
        Statistics:
        Mean R: {np.mean(self.R_values):.3f}
        Max R: {np.max(self.R_values):.3f}
        R>1: {np.sum(self.R_values > 1.0)} ({np.mean(self.R_values > 1.0)*100:.1f}%)
        """
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ---- 右子图: 奇异值谱 ----
        ax2.plot(k_index, sigma_np[:d], 'g-', linewidth=2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Principal Component Index (k)', fontsize=12)
        ax2.set_ylabel('Singular Value (σ_k)', fontsize=12)
        ax2.set_title('Singular Value Spectrum (σ_k)', fontsize=14, fontweight='bold')
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        
        # 添加奇异值信息
        svd_text = f"""
        Singular Values:
        Max σ: {sigma_np[0]:.3e}
        Min σ: {sigma_np[d-1]:.3e}
        Condition: {sigma_np[0]/sigma_np[d-1]:.3e}
        """
        ax2.text(0.02, 0.98, svd_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.suptitle('Activation Perturbation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        if save_fig:
            fig_path = os.path.join(self.output_dir, fig_name)
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            print(f"分析图表已保存到: {fig_path}")
            plt.close(fig)  # 关闭图形释放内存
            return fig_path
        else:
            plt.show()
            return fig
    
    def run_full_analysis(self, 
                         save_fig: bool = True,
                         save_results: bool = True,
                         save_data: bool = False) -> Dict:
        """
        运行完整分析流程
        
        Args:
            save_fig: 是否保存图表
            save_results: 是否保存分析结果
            save_data: 是否保存原始数据
            
        Returns:
            分析结果字典
        """
        print("="*60)
        print("开始激活值分析")
        print(f"输出目录: {self.output_dir}")
        print("="*60)
        
        # 检查数据是否已加载
        if self.sigma_original is None or self.U_o is None or self.h_orig is None or self.h_edit is None:
            print("警告: 数据未完全加载，尝试自动加载...")
            self.load_all_data()
        
        # 执行分析流程
        self.compute_perturbation()
        self.compute_R_values()
        stats = self.analyze_R_statistics()
        
        # 保存输出
        saved_files = {}
        
        if save_fig:
            fig_path = self.plot_analysis(save_fig=True)
            saved_files['figure'] = fig_path
        
        if save_results:
            json_path = self.save_results(stats)
            saved_files['json'] = json_path
        
        if save_data:
            npz_path = self.save_R_values()
            saved_files['npz'] = npz_path
        
        print("\n" + "="*60)
        print("激活值分析完成！")
        print("="*60)
        
        return {
            'statistics': stats,
            'R_values': self.R_values,
            'saved_files': saved_files
        }


if __name__ == "__main__":    
    analyzer = ActivationAnalyzer(output_dir='/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/wyren/results')
    
    analyzer.load_json_data('/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/xsong/last/Llama-3.1-8B-Instruct_ROME_num1_layer_30.json')
    analyzer.load_npz_data('/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/xsong/last/Llama-3.1-8B-Instruct_ROME_num1_layer_30_matrices.npz')
    analyzer.load_pt_data(
        edit_path='/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/xsong/last/hidden_states/layer_30.pt',
        orig_path='/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/xsong/last/hidden_states/rome_num1_layer_30.pt'
    )
    
    # 分步分析
    analyzer.compute_perturbation()
    analyzer.compute_R_values()
    stats = analyzer.analyze_R_statistics()
    
    # 自定义保存
    analyzer.plot_analysis(
        save_fig=True,
        fig_name="custom_analysis_plot.png",  
        dpi=150
    )
    analyzer.save_results(stats, "custom_stats.json")