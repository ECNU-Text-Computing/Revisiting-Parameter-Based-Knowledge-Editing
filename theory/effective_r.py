import numpy as np
import torch

# 假设 H_centered 尺寸 (4096, 1000)
H = H_centered  # d=4096, n=1000

# 做 SVD，因为 n<d，用 full_matrices=True 得到完整的 U (4096,4096)
U, S, Vh = torch.linalg.svd(H, full_matrices=True)  # S 长度 = min(d,n) = 1000
S = S.cpu().numpy()  # 长度 1000

# 方法1：95% 能量阈值
energy = S**2
cum_energy_ratio = np.cumsum(energy) / np.sum(energy)
r_95 = np.argmax(cum_energy_ratio >= 0.95) + 1  # 1-based 索引

# 方法2：奇异值相对最大值的 1% 阈值
r_1percent = np.sum(S >= 0.01 * S[0])

# 方法3：自动拐点检测（简化版）
# 计算对数奇异值的二阶差分
logS = np.log(S + 1e-12)
diff2 = np.diff(logS, n=2)  # 二阶差分
# 找最大的负二阶差分位置（下降最快处）
elbow = np.argmin(diff2) + 2  # 二阶差分比原数组短2

print(f"基于95%能量: r = {r_95}")
print(f"基于1%阈值: r = {r_1percent}")
print(f"拐点法: r ≈ {elbow}")