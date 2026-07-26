# todo
选用的数据集是哪个
怎么设计函数抓取需要的指标并存储到指定位置？
    hook函数
怎么计算奇异值分解
![alt text](image.png)


# 路径
```python
# 模型路径
llama3.1-8b: '/home/wyren/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'


# 数据集路径
zsre: '/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/zsre/ZsRE-test-all-sentence_formatted.json'

```

```bash
hf download google-t5/t5-base
(easyedit2) wyren@DGX-A800:~/Knowledge-Editing-Benchmark$ conda list transformer
# packages in environment at /home/wyren/miniconda3/envs/easyedit2:
#
# Name                           Version          Build            Channel
sentence-transformers            5.1.2            pypi_0           pypi
transformer-lens                 2.16.1           pypi_0           pypi
transformers                     4.57.0           pypi_0           pypi
transformers-stream-generator    0.0.5            pypi_0           pypi
(easyedit2) wyren@DGX-A800:~/Knowledge-Editing-Benchmark$ conda list peft
# packages in environment at /home/wyren/miniconda3/envs/easyedit2:
#
# Name                     Version          Build            Channel
peft                       0.18.0           pypi_0           pypi
(easyedit2) wyren@DGX-A800:~/Knowledge-Editing-Benchmark$ conda list sentence-tran
# packages in environment at /home/wyren/miniconda3/envs/easyedit2:
#
# Name                     Version          Build            Channel
sentence-transformers      5.1.2            pypi_0           pypi
```

# 实验指标

严格区分了**直接观测**（代码里直接 `save`的变量）和**后期计算**（不需要在实验循环中保存）。

### 一、编辑前（原始模型）的原始指标

在**未编辑的原始模型**上，你只需要跑一遍数据，收集以下最基础的表示向量：

| 指标名称                      | 数据结构                        | 保存格式建议          | 备注                                                         |
| ----------------------------- | ------------------------------- | --------------------- | ------------------------------------------------------------ |
| **`h_original`**              | `torch.Tensor`of shape `[n, d]` | `.pt`(PyTorch tensor) | **这是最重要的基石**。`n=2000`个样本在目标层的输出向量。     |
| **`x_input_original`** (可选) | `torch.Tensor`of shape `[n, d]` | `.pt`                 | 该层的输入向量。用于后续可能的 Jacobian 近似分析。           |
| **`a_original`** (可选)       | `torch.Tensor`of shape `[n, d]` | `.pt`                 | 激活函数前的线性输出（`W * x`）。同样用于 Jacobian 分析。    |
| **`prompt_ids`**              | List of `n`strings              | `.json`               | 对应的 2000 条 prompt 文本或 token ids，确保后续能复现输入。 |

> **注意**：编辑前**不需要**计算 SVD、奇异值或主成分，这些全是**后期计算**。

------

### 二、编辑后（每次编辑后）的原始指标

对于**第 T 次编辑**（T=1,2,...），在**同一批输入**上运行，收集以下对比数据：

| 指标名称                | 数据结构                        | 保存格式建议 | 备注                                                         |
| ----------------------- | ------------------------------- | ------------ | ------------------------------------------------------------ |
| **`h_edited`**          | `torch.Tensor`of shape `[n, d]` | `.pt`        | 编辑后模型在**同一层**的输出。                               |
| **`delta_h`**           | `torch.Tensor`of shape `[n, d]` | `.pt`        | **必须保存**。`delta_h = h_edited - h_original`（向量差）。  |
| **`delta_h_norm`**      | `torch.Tensor`of shape `[n]`    | `.pt`        | 每个样本扰动的 L2 范数 `\|Δh_i\|`。                          |
| **`W_edited`** (若可行) | `torch.Tensor`of shape `[d, d]` | `.pt`        | 编辑后的权重矩阵（如 ROME 修改后的 `W`）。用于计算 `\|ΔW\|`。 |
| **`edit_metadata`**     | Dict                            | `.json`      | 记录编辑事实、编辑方法、时间戳、编辑次数 `T`。               |

------

### 三、后期计算指标（非原始数据，勿在实验循环中保存）

这些指标利用上述原始数据**离线计算**，不应占用实验运行时的存储和 I/O：

| 计算指标                | 计算公式/方法                                                | 用途                       |
| ----------------------- | ------------------------------------------------------------ | -------------------------- |
| **奇异值谱 `σ_orig`**   | `torch.linalg.svdvals(H_original.T)`                         | 验证维度坍缩（H4.1）       |
| **主成分方向 `U_orig`** | `U, S, Vh = torch.linalg.svd(H_original.T, full_matrices=True)` | 建立原始空间的坐标系       |
| **投影系数 `c_k_i`**    | `C = U_orig.T @ delta_h.T`(矩阵乘法)                         | 计算扰动在原始方向上的分量 |
| **经验放大因子 `R_k`**  | `R_k = (√n * mean(\|c_k_i\|)) / σ_k`                         | 验证扰动放大（定理 4.6）   |
| **主成分稳定性**        | 比较 `U_orig`和 `U_edit`的子空间夹角                         | 评估假设 H4.3 的违背程度   |

### 四、文件目录结构建议

建议按以下方式组织文件：

```
exp_data/
├── 00_original/          # 编辑前基准
│   ├── h_original.pt
│   ├── prompts.json
│   └── config.yaml
├── edit_01/              # 第 1 次编辑
│   ├── h_edited.pt
│   ├── delta_h.pt
│   ├── delta_h_norm.pt
│   └── metadata.json
├── edit_05/              # 第 5 次编辑
│   ├── ...
```

**一句话总结**：你只需要在代码里循环保存 `h_original`、`h_edited`和 `delta_h`这三个张量，其余所有分析（SVD、投影、R_k）全部留到实验跑完后再做。这样既保证了数据完整性，又最大程度简化了实验运行逻辑。