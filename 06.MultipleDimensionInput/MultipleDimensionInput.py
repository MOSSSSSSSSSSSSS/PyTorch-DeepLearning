# ①比如说一个二分类的dataset，由输入x和输出y组成，共n个样本，x有8列，y输出就是1列，列就是x和y的特征。可以把x都提取出来，是一个n行8列的矩阵
#   y提取出来是一个n行1列的矩阵
# ②上一个例子是一个维度（一列）的x输入，这个是多维度的x，第i行y_pred的计算也会有改变，是σ(Σ(xj * wj) + b)(有8列，j就从1到8)，
#   为了更好地发挥gpu并行计算的能力，可以直接将x那个n*8的矩阵乘w 8*1的矩阵（而不是for循环挨个算，会很慢），再加n*1的b矩阵（同一个b复制n次），得到n*1的z矩阵，
#   再用σ(z矩阵)函数得到y_pred矩阵，pytorch里的函数包括σ()函数是向量式的函数，是将n*1的z矩阵的每一个元素单独用σ函数计算，再放到n*1的y_pred矩阵中
# ③可以不只加一层线性变换，可以加好几层，每层之间都要加一个非线性的函数变换，8维可以先降到6维，6维再降到2维，再降到1维，也可以中间维度上升
#   取多少层、中间如何取值，是个典型的超参数的搜索，一般隐层越多，学习能力越强，但学习能力太强，会学习到噪声的规律，泛化能力降低
# ④神经网络用32位浮点数就够用了，一般的游戏的显卡如1080等只支持32位浮点数运算，很贵的英伟达特斯拉系列才支持double类型的数据
# ⑤gz是linux下的压缩格式
import numpy as np
import torch

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)  # delimiter分隔符，这个文件包含了x和y
x_data = torch.from_numpy(xy[:, :-1])   # 除了-1那一行都要
y_data = torch.from_numpy(xy[:, [-1]])  # -1要加中括号目的是y_data是个矩阵而不是向量


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(8, 6)
        self.Linear2 = torch.nn.Linear(6, 4)
        self.Linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()  # 上次是torch.nn.functional的sigmoid函数，这次是模块，可以作为一个层，这个模块没有参数，只需一个
# 也叫激活函数，这里也可以改成其他函数，改成Relu函数，relu函数取值0到1，可以是0，但在计算BCEloss是可能有对数出现，所以这里可以换成relu函数，但最后的激活函数
# 应写成sigmoid函数

    def forward(self, x):
        x = self.activate(self.Linear1(x))
        x = self.activate(self.Linear2(x))
        x = self.activate(self.Linear3(x))
        return x


model = Model()

criterion = torch.nn.BCEloss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)  # 没有用mini-batch，we shall talk about Dataloader later
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


