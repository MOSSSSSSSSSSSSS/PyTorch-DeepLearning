# 一个5-*1的矩阵变成一个6*1的矩阵，需要一个6*5的矩阵，里面有30个权重，因此一个神经网络权重是很多的，可以用反向传播更简便地计算loss对权重的偏导,从而通过随机梯度下降让loss越来越小
# bias偏差 y = W * x + b，bias偏差，为了增加复杂度，y要经过一个非线性的函数的变换，在进入下一层，否则再多层都能通过一层表示出来
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # 一维张量，先随机为1.0，张量数据结构包括data，和grad，grad是梯度，也是张量，张量的运算会自动转换类型为张量并构建计算图
w.requires_grad = True   # 需要梯度

def forward(x):
    return x * w         # w是张量，x不是，会转换为张量并构建计算图

def loss(x, y):
    return (forward(x) - y) ** 2  # 构建计算图

for epoch in range(100): # 0-99
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()     # 反向传播计算需要梯度的张量，即计算l对于w的偏导，执行这个函数后会使构建的计算图消失
        print('\tgrad:', x, y, w.grad.item())
        w.data -= 0.01 * w.grad.data   # 或w.grad.item()  # 要用data计算而不能用张量计算，否则会构建一直构建，构建一个很长计算图使内存耗尽

        w.grad.data.zero_()       # 要清零，否则 每次算出来的 梯度会累加 ，而我们不想累加
    print("progress:", epoch, l.item())
print("predict (after training)", 4.0 forward(4.0).item())
