import numpy as np
import matplotlib.pyplot as plt

# 定义参数
A = 20.41
a = 9.03
b = 13.99
R_sun = 8.3
R_paf = 3.76
r_max = 20  # 设定 r 的取值上限，可根据实际情况调整

# 计算目标函数在区间 [0, r_max] 上的最大值 M
r_vals = np.linspace(0, r_max, 10000)
rho_vals = A * ((r_vals + R_paf) / (R_sun + R_paf)) ** a * np.exp(-b * (r_vals - R_sun) / (R_sun + R_paf))
M = np.max(rho_vals)

# 拒绝采样生成样本
samples = []
while len(samples) < 130000:
    r = np.random.uniform(0, r_max)
    u = np.random.uniform(0, 1)
    rho_r = A * ((r + R_paf) / (R_sun + R_paf)) ** a * np.exp(-b * (r - R_sun) / (R_sun + R_paf))
    if u <= rho_r / M:
        samples.append(r)
samples = np.array(samples)

# 绘制理论曲线和样本直方图
plt.figure(figsize=(10, 6))
plt.plot(r_vals, rho_vals, label='Theoretical Curve', linewidth=2)
plt.hist(samples, bins=100, density=True, label='Sample Histogram', alpha=0.7)
plt.xlabel('r')
plt.ylabel('Density')
plt.title('Rejection Sampling for Pulsar Surface Density')
plt.legend()
plt.savefig('pulsar_sampling_result.png')
plt.show()