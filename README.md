GitHub地址：
代码详解：
该代码使用拒绝采样法生成服从特定密度分布 ρ(r) 的随机样本，并通过可视化验证采样结果与理论分布的一致性。
import numpy as np
import matplotlib.pyplot as plt
# 定义参数
A = 20.41       # 振幅系数
a = 9.03        # 多项式指数
b = 13.99       # 指数衰减系数
R_sun = 8.3     # 太阳位置半径（kpc）
R_paf = 3.76    # 特征半径（kpc）
r_max = 20      # 采样上限半径
作用：设置天体物理参数和采样范围

# 计算目标函数在区间 [0, r_max] 上的最大值 M
r_vals 
= np.linspace(0, r_max, 10000)
rho_vals 
= A * ((r_vals + R_paf)/(R_sun + R_paf))**a * np.exp(-b*(r_vals - R_sun)/(R_sun + R_paf))
M 
= np.max(rho_vals)

关键操作：

np.linspace(0, r_max, 10000)：在 [0, 20] 区间生成等间距的10,000个点

计算所有点的密度值 ρ(r)

np.max(rho_vals)：找到最大密度值用于归一化

# 拒绝采样生成样本
samples = []
while len(samples) < 130000:
    r = np.random.uniform(0, r_max)  # 从均匀分布采样
    u = np.random.uniform(0, 1)      # 生成接受阈值
    # 计算当前点的密度
    rho_r = A * ((r + R_paf)/(R_sun + R_paf))**a * np.exp(-b*(r - R_sun)/(R_sun + R_paf))
    if u <= rho_r / M:  # 接受条件判断
        samples.append(r)
samples = np.array(samples)

1、拒绝采样逻辑：
2、生成候选样本 r ~ U(0, 20)
3、计算接受概率 ρ(r)/M
4、通过均匀随机数 u 决定是否接受样本


参数	影响范围	推荐值范围
r_max	采样范围上限	20-30 kpc
bins	直方图精细度	50-200
n_samples	采样数量	>10,000
