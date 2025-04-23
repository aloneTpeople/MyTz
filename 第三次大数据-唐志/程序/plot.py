import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegressionCV


# 设置随机种子和生成数据
np.random.seed(10)
X, y = make_moons(200, noise=0.20)
# 自定义颜色和形状
colors = ['CornflowerBlue', 'Tomato']     # 蓝、橙
markers = ['o', '*']                # o: 圆形, s: 正方形
plt.figure(figsize=(8, 6))
for i in range(2):  # 类别 0 和 1
    plt.scatter(X[y==i, 0], X[y==i, 1],
                s=50,
                c=colors[i],
                marker=markers[i],
                label=f'Class {i}',
                edgecolors='None',
                alpha=0.8)
plt.title("Customized Scatter Plot of make_moons")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def plot_decision_boundary(pred_func):
    # 设置边界范围和网格间隔
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #首先将变量X的所有行中第一列（索引为 0）的最小值减去 0.5 后赋值给x_min，
    # 将变量X的所有行中第一列（索引为 0）的最大值加上 0.5 后赋值给x_max。
    # 接着将变量X的所有行中第二列（索引为 1）的最小值减去 0.5 后赋值给y_min，
    # 将变量X的所有行中第二列（索引为 1）的最大值加上 0.5 后赋值给y_max。
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 网格点的预测
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ===== 自定义填充颜色 =====
    cmap_background = ListedColormap(['#a0c4ff', '#ffc9c9'])  # 浅蓝+浅橙
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)

    # ===== 自定义点颜色和形状 =====
    colors = ['CornflowerBlue', 'Tomato']
    markers = ['o', '*']
    for i in range(2):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    s=60,
                    c=colors[i],
                    marker=markers[i],
                    label=f'Class {i}',
                    edgecolors='None',
                    alpha=0.9)


clf = LogisticRegressionCV()
clf.fit(X, y)
plt.figure(figsize=(8, 6))
# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.savefig("ai_net_img_02.png", dpi=300)
plt.show()




