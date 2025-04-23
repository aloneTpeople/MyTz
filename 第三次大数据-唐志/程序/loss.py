from plot import *
import numpy as np
import matplotlib.pyplot as plt


# --------------- 定义signoid激活函数 ---------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------- 定义全局超参数 ---------------

# 模型和训练参数
num_examples = len(X)      # 样本数量
nn_input_dim = 2           # 输入层维度（数据是二维坐标）
nn_output_dim = 2          # 输出层维度（二分类问题）
epsilon = 0.01             # 学习率
reg_lambda = 0.01          # L2 正则化系数


# 计算整个数据集上的总损失（用于评估模型效果）
def calculate_loss(model):
    # 从模型中提取参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算预测概率
    z1 = X.dot(W1) + b1                   # 输入层 → 隐藏层
    a1 = sigmoid(z1)                      # 激活函数：signoid
    z2 = a1.dot(W2) + b2                  # 隐藏层 → 输出层
    exp_scores = np.exp(z2)               # 对每个类别计算 e^score
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布

    # 计算交叉熵损失（对数损失）
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)

    # 加入 L2 正则化项（防止过拟合）
    data_loss += (reg_lambda / 2) * (
        np.sum(np.square(W1)) + np.sum(np.square(W2))
    )

    # 返回平均损失
    return data_loss / num_examples
# 预测函数：根据输入样本 x，输出类别（0 或 1）
def predict(model, x):
    # 解包模型参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算每个类别的概率
    z1 = x.dot(W1) + b1          # 输入层 → 隐藏层
    a1 =sigmoid(z1)             # 激活函数：sigmoid
    z2 = a1.dot(W2) + b2         # 隐藏层 → 输出层
    exp_scores = np.exp(z2)      # 指数函数
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布

    # 返回每个样本概率最大的类别索引（即预测结果）
    return np.argmax(probs, axis=1)
# 训练神经网络，学习模型参数并返回最终模型
# 参数说明：
# - nn_hdim：隐藏层的节点数量
# - num_passes：迭代次数（训练轮数）
# - print_loss：是否每 1000 次打印一次损失
def build_model(nn_hdim, num_passes=30000, print_loss=False):
    np.random.seed(0)

    # 参数初始化（权重随机初始化 + 偏置初始化为 0）
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    # 训练过程：使用全量批量梯度下降（Batch GD）
    for i in range(num_passes):
        # -------- 前向传播 --------
        z1 = X.dot(W1) + b1               # 输入层 → 隐藏层
        a1 =sigmoid(z1)                  # 激活函数：sigmoid
        z2 = a1.dot(W2) + b2              # 隐藏层 → 输出层
        exp_scores = np.exp(z2)           # softmax 的分子部分
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率

        # -------- 反向传播 --------
        delta3 = probs
        delta3[range(num_examples), y] -= 1         # 输出误差（预测 - 真实）
        dW2 = a1.T.dot(delta3)                      # 输出层权重梯度
        db2 = np.sum(delta3, axis=0, keepdims=True) # 输出层偏置梯度

        delta2 = delta3.dot(W2.T) * (a1 * (1 - a1))  # signoid 的导数：a1 * (1 - a1)
        dW1 = X.T.dot(delta2)                             # 隐藏层权重梯度
        db1 = np.sum(delta2, axis=0)                      # 隐藏层偏置梯度

        # -------- 正则化（L2）--------
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # -------- 参数更新（梯度下降）--------
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        # 保存更新后的参数
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 每 1000 次输出一次损失值（可选）
        if print_loss and i % 1000 == 0:
            print(f"迭代 {i} 次后的损失值：{calculate_loss(model):.6f}")

    return model
# 构建一个隐藏层维度为 3 的神经网络模型，并训练
model = build_model(nn_hdim=3, print_loss=True)
plt.figure(figsize=(8, 6))
# 使用训练好的模型绘制决策边界
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.savefig("ai_net_img_03.png",dpi=300)
plt.show()


# 可视化不同隐藏层节点数对模型决策边界的影响
plt.figure(figsize=(18, 35))  # 设置整体图像大小
plt.subplots_adjust(
    left=0.08,   # 左侧边距压缩（原图默认0.125）
    right=0.95,  # 右侧边距压缩
    bottom=0.08, # 底部边距压缩（留足X轴标签空间）
    top=0.92,    # 顶部边距压缩（留足标题空间）
    wspace=0.4,  # 水平间距（保持原调整）
    hspace=0.5   # 垂直间距（略减，因画布已增大）
)
# 隐藏层节点数量列表
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]

# 遍历不同隐藏层大小，训练模型并绘图
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)  # 创建子图：5 行 2 列，第 i+1 个

    plt.title(f"Hidden Layer size: {nn_hdim}")  # 设置子图标题
    model = build_model(nn_hdim)  # 训练模型
    plot_decision_boundary(lambda x: predict(model, x))  # 绘制决策边界
    plt.xlabel("X1",labelpad=10)# 标签与轴间距加大
    plt.ylabel("X2",labelpad=10)

# 显示所有子图
#plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.savefig("ai_net_img_04.png",dpi=300)
plt.show()


