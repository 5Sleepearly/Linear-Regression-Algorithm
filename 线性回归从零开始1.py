# _*_ coding:utf-8 _*_
# Author: WangYang
# CreaTime: 2022/8/26
# FileName: 线性回归从零开始1
# Description: simple introduction of the code

import random
import torch
from d2l import torch as d2l


# 根据带有噪声的线性模型构造一个人造数据集。我们使用线性模型参数，w是权重，b，e是噪声
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 均值为0，方差为1，个数为num_examples，列数为长度
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # -1自动计算，一代表一列


true_w = torch.tensor([2, -3.4])  # 输入w b
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # 调用函数，生成特征和标注

print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)  # 画出，特征的第一列


# 定义一个函数，实现读取小批量数据，特征矩阵和标签向量作为输入，生成大小为batch_size
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 打乱顺序
    for i in range(0, num_examples, batch_size):  # rang 的使用，步长
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])  # 生成tensor
        yield features[batch_indices], labels[batch_indices]  # 相当于return


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):  # @save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # @save   #预测值和真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # @save   #params是所有参数是一个list，lr学习率
    """小批量随机梯度下降"""
    with torch.no_grad():  # 不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 手动设置梯度为0


# 训练过程
lr = 0.001  # 指定超参数学习率为0.03
num_epochs = 10  # 指定超参数，把整个数据扫描三遍
net = linreg  # 指定超参数，定义线性模型
loss = squared_loss  # 指定超参数，均方损失

for epoch in range(num_epochs):  # 每一次对数据扫一遍
    for X, y in data_iter(batch_size, features, labels):  # 对于每一次拿出批量大小的x,y
        l = loss(net(X, w, b), y)  # X和y的小批量损失，把s,w,b放进net里面做预测，预测的y与真实y做损失
        # 损失是一个向量
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()  # 求和，求和之后，反向传播算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 比较真实参数和通过训练学到的参数来评估训练的成功程度
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
