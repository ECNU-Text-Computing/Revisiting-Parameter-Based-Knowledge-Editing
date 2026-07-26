import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple, List, Dict, Optional, Union
from matplotlib import cm

class PrincipalAngleAnalyzer:
    """主角度分析器，用于分析两个子空间之间的稳定性"""
    
    def __init__(self, 
                 file_path: str,
                 output_dir: str = "./results",
                 k_list: Optional[List[int]] = None):
        """
        初始化分析器
        
        Args:
            file_path: .npz文件路径，包含U_original和U_edited矩阵
            output_dir: 输出目录，用于保存图片和结果文件
            k_list: 分析的子空间维度列表，默认为[10, 50, 100, 200, 500, 1000, 2000]
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.k_list = k_list or [10, 50, 100, 200, 500, 1000, 2000]
        
        self.U_o = None
        self.U_e = None
        self.results = {}
        self.principal_angles = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self) -> None:
        """加载数据"""
        data = np.load(self.file_path)
        self.U_o = torch.from_numpy(data['U_original']).float()
        self.U_e = torch.from_numpy(data['U_edited']).float()
        print(f"数据加载完成: U_original.shape={self.U_o.shape}, U_edited.shape={self.U_e.shape}")
        
    @staticmethod
    def compute_principal_angles_stable(U1: torch.Tensor, U2: torch.Tensor, k: int) -> np.ndarray:
        """
        稳健地计算主角度余弦值
        
        Args:
            U1, U2: 输入矩阵 (d, n)
            k: 子空间维度
            
        Returns:
            主角度余弦值数组
        """
        U1_k = U1[:, :k]  # (d, k)
        U2_k = U2[:, :k]  # (d, k)
        
        # QR分解提高稳定性
        Q1, R1 = torch.linalg.qr(U1_k)
        Q2, R2 = torch.linalg.qr(U2_k)
        
        # 计算 M = Q1^T Q2
        M = Q1.T @ Q2
        
        # 对M做SVD
        try:
            cos_theta = torch.linalg.svd(M, full_matrices=False)[1]
        except torch.linalg.LinAlgError:
            cos_theta = torch.linalg.svd(M, driver='gesvd')[1]
        
        # 确保余弦值在[0,1]范围内
        cos_theta = torch.clamp(cos_theta, 0, 1)
        return cos_theta.cpu().numpy()
    
    def compute_all_angles(self) -> None:
        """计算所有k值的主角度"""
        if self.U_o is None or self.U_e is None:
            raise ValueError("请先调用load_data()加载数据")
            
        self.principal_angles = {}
        for k in self.k_list:
            if k > min(self.U_o.shape[1], self.U_e.shape[1]):
                print(f"警告: k={k} 超出矩阵维度，跳过计算")
                continue
                
            cos_theta = self.compute_principal_angles_stable(self.U_o, self.U_e, k)
            self.principal_angles[k] = cos_theta
            print(f"计算完成: k={k}, 得到 {len(cos_theta)} 个主角度余弦值")
    
    def analyze_results(self) -> None:
        """分析主角度结果"""
        if not self.principal_angles:
            raise ValueError("请先调用compute_all_angles()计算主角度")
            
        self.results = {}
        print("\n=== 主角度余弦值分析 ===")
        
        for k, cos_theta in self.principal_angles.items():
            cos_theta = np.array(cos_theta)
            self.results[k] = {
                'mean': np.mean(cos_theta),
                'min': np.min(cos_theta),
                'max': np.max(cos_theta),
                'median': np.median(cos_theta),
                'std': np.std(cos_theta),
                'last_5_mean': np.mean(cos_theta[-5:]),  # 最后5个（最小主角度）
                'last_1': cos_theta[-1],  # 最小主角度的余弦值
                'num_angles': len(cos_theta)
            }
            
            print(f"k={k}:")
            print(f"  均值={self.results[k]['mean']:.6f}, 中位数={self.results[k]['median']:.6f}")
            print(f"  最小值={self.results[k]['min']:.6f} (最小主角度余弦)")
            print(f"  标准差={self.results[k]['std']:.6f}")
            print(f"  最后5个均值={self.results[k]['last_5_mean']:.6f}")
            print()
    
    def save_results(self, filename: str = "principal_angles_results.json") -> str:
        """
        保存分析结果到JSON文件
        
        Args:
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if not self.results:
            raise ValueError("没有可保存的结果，请先运行分析")
            
        # 准备可序列化的结果
        serializable_results = {}
        for k, data in self.results.items():
            serializable_results[str(k)] = {key: (float(value) if isinstance(value, (np.float32, np.float64)) else value)
                                           for key, value in data.items()}
            
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'file_path': self.file_path,
                'k_list': self.k_list,
                'analysis_time': np.datetime64('now').astype(str),
                'matrix_shapes': {
                    'U_original': list(self.U_o.shape) if self.U_o is not None else None,
                    'U_edited': list(self.U_e.shape) if self.U_e is not None else None
                },
                'results': serializable_results
            }, f, indent=2, ensure_ascii=False)
            
        print(f"结果已保存到: {file_path}")
        return file_path
    
    def save_angles_data(self, filename: str = "principal_angles_data.npz") -> str:
        """
        保存主角度数据到.npz文件
        
        Args:
            filename: 输出文件名
            
        Returns:
            保存的文件路径
        """
        if not self.principal_angles:
            raise ValueError("没有可保存的主角度数据，请先运行计算")
            
        file_path = os.path.join(self.output_dir, filename)
        
        # 准备数据
        save_data = {}
        for k, cos_theta in self.principal_angles.items():
            save_data[f'k_{k}'] = cos_theta
            
        np.savez(file_path, **save_data)
        print(f"主角度数据已保存到: {file_path}")
        return file_path
    
    def plot_results(self, 
                    save_fig: bool = True,
                    fig_name: str = "principal_angles_analysis.png",
                    dpi: int = 300,
                    figsize: Tuple[int, int] = (12, 5)) -> Optional[plt.Figure]:
        """
        绘制分析结果
        
        Args:
            save_fig: 是否保存图片
            fig_name: 图片文件名
            dpi: 图片分辨率
            figsize: 图片尺寸
            
        Returns:
            matplotlib图形对象，如果save_fig为False
        """
        if not self.results:
            raise ValueError("请先调用analyze_results()分析结果")
            
        k_list = [k for k in self.k_list if k in self.results]  # 只使用有结果的数据
        min_cos_vals = [self.results[k]['min'] for k in k_list]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：最小cosθ
        ax1.plot(k_list, min_cos_vals, 'o-', markersize=8, linewidth=2, color='blue')
        ax1.set_xlabel('Subspace Dimension k', fontsize=12)
        ax1.set_ylabel('Cosine of Smallest Principal Angle', fontsize=12)
        ax1.set_title('Minimum Principal Angle Cosine\n(closer to 1 = more stable)', fontsize=13, fontweight='bold')
        ax1.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label='Excellent (θ<8°)')
        ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Fair (θ<26°)')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Poor (θ>60°)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # 标记突变点
        for k, cos_val in zip(k_list, min_cos_vals):
            if cos_val < 0.1:  # 角度>84°
                ax1.text(k, cos_val+0.05, f'{cos_val:.3f}', 
                        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            elif cos_val < 0.9:  # 角度>26°
                ax1.text(k, cos_val+0.05, f'{cos_val:.3f}', 
                        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        
        # 子图2：夹角变化
        min_angles_deg = np.degrees(np.arccos(np.clip(min_cos_vals, 0, 1)))  # 转为角度
        ax2.plot(k_list, min_angles_deg, 's-', markersize=8, linewidth=2, color='red')
        ax2.set_xlabel('Subspace Dimension k', fontsize=12)
        ax2.set_ylabel('Smallest Angle (degrees)', fontsize=12)
        ax2.set_title('Minimum Principal Angle (degrees)\n(smaller = more stable)', fontsize=13, fontweight='bold')
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Excellent (<10°)')
        ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Fair (<30°)')
        ax2.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Poor (>60°)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        
        # 保存图片
        if save_fig:
            fig_path = os.path.join(self.output_dir, fig_name)
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            print(f"分析图表已保存到: {fig_path}")
            plt.close(fig)  # 关闭图形释放内存
        else:
            plt.show()
            return fig
    
    def print_interpretation(self, save_txt: bool = True, txt_name: str = "interpretation.txt") -> str:
        """
        打印角度解释
        
        Args:
            save_txt: 是否保存到文本文件
            txt_name: 文本文件名
            
        Returns:
            保存的文本文件路径（如果保存）
        """
        if not self.results:
            raise ValueError("请先调用analyze_results()分析结果")
            
        k_list = [k for k in self.k_list if k in self.results]
        min_cos_vals = [self.results[k]['min'] for k in k_list]
        
        # 构建输出内容
        output_lines = []
        output_lines.append("="*60)
        output_lines.append("最小主角度解释")
        output_lines.append("="*60)
        output_lines.append(f"数据文件: {os.path.basename(self.file_path)}")
        output_lines.append(f"分析时间: {np.datetime64('now').astype(str)}")
        output_lines.append("")
        
        for k, cos_val in zip(k_list, min_cos_vals):
            angle_deg = np.degrees(np.arccos(max(cos_val, 1e-10)))
            
            if angle_deg < 10:
                stability = "高度稳定"
            elif angle_deg < 30:
                stability = "基本稳定" 
            elif angle_deg < 60:
                stability = "中度不稳定"
            elif angle_deg < 80:
                stability = "严重不稳定"
            else:
                stability = "几乎正交（极端不稳定）"
            
            line = f"k={k:4d}: cosθ_min={cos_val:.6f}, θ_min={angle_deg:6.1f}° → {stability}"
            output_lines.append(line)
        
        # 汇总统计
        output_lines.append("\n" + "="*60)
        output_lines.append("稳定性汇总")
        output_lines.append("="*60)
        
        stability_counts = {
            "高度稳定": 0,
            "基本稳定": 0,
            "中度不稳定": 0,
            "严重不稳定": 0,
            "几乎正交（极端不稳定）": 0
        }
        
        for k, cos_val in zip(k_list, min_cos_vals):
            angle_deg = np.degrees(np.arccos(max(cos_val, 1e-10)))
            
            if angle_deg < 10:
                stability_counts["高度稳定"] += 1
            elif angle_deg < 30:
                stability_counts["基本稳定"] += 1
            elif angle_deg < 60:
                stability_counts["中度不稳定"] += 1
            elif angle_deg < 80:
                stability_counts["严重不稳定"] += 1
            else:
                stability_counts["几乎正交（极端不稳定）"] += 1
        
        for stability, count in stability_counts.items():
            output_lines.append(f"{stability}: {count} 个维度")
        
        # 输出到控制台
        for line in output_lines:
            print(line)
        
        # 保存到文本文件
        if save_txt:
            txt_path = os.path.join(self.output_dir, txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n解释文本已保存到: {txt_path}")
            return txt_path
        
        return None
    
    def run_full_analysis(self, 
                          save_fig: bool = True,
                          save_results: bool = True,
                          save_angles: bool = False) -> Dict:
        """
        运行完整分析流程
        
        Args:
            save_fig: 是否保存图表
            save_results: 是否保存分析结果
            save_angles: 是否保存原始角度数据
            
        Returns:
            分析结果字典
        """
        print("="*60)
        print("开始主角度分析")
        print(f"数据文件: {self.file_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"分析维度: {self.k_list}")
        print("="*60)
        
        # 执行分析流程
        self.load_data()
        self.compute_all_angles()
        self.analyze_results()
        
        # 保存输出
        saved_files = {}
        
        if save_fig:
            fig_path = self.plot_results(save_fig=True)
            if fig_path:
                saved_files['figure'] = fig_path
        
        if save_results:
            json_path = self.save_results()
            saved_files['json'] = json_path
        
        if save_angles:
            npz_path = self.save_angles_data()
            saved_files['npz'] = npz_path
        
        txt_path = self.print_interpretation(save_txt=True)
        if txt_path:
            saved_files['txt'] = txt_path
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        
        return {
            'results': self.results,
            'saved_files': saved_files
        }

if __name__ == "__main__":
    analyzer = PrincipalAngleAnalyzer(
        file_path='/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/xsong/last/Llama-3.1-8B-Instruct_ROME_num1_layer_30_matrices.npz',
        output_dir='/home/wyren/Knowledge-Editing-Benchmark/wyren/supplementary_exp/wyren/results',
        k_list=[10, 50, 100, 200, 500, 1000, 2000, 3000, 4000]  # 自定义k值
    )
    
    result = analyzer.run_full_analysis(save_fig=True, save_results=True, save_angles=True)
    