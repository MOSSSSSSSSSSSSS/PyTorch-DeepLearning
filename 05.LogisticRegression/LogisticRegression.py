# mnist数据集是一个手写数字的数据集，有60000个训练集，10000个测试集，分类为10即0,1,2,3,4,5,6,7,8,9
# CIFAR-10 dataset是一个图片的数据集，分为10类，有airplane类，bird类，cat类等，50000训练集10000测试集
# 有两个分类的问题叫二分类
# Logistic函数是最常用的sigmoid函数，有时就把它叫做sigmoid函数，它写作 σ(x)，取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间，可用来做二分类
# logistic regression是分类用的，分类问题输出的是  属于每一个分类的概率，取概率最高的
# logistic回归单元，相比线性回归单元，就是将回归单元的输出放到σ(x)中再输出
# 求loss，二分类里的损失函数叫BCE（二分类交叉熵） loss = -(ylog(y_pred) + (1 - y)log(1 - y_pred)),y只可能是1或0，y_pred属于(0,1)，这是一组数据的loss，mini-batch
# 的是好几组数据的loss相加求平均值，优化器依然是让loss更小
#
#import torchvision
#train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)  # 存放位置，是不是训练集，从不从网上下载（已有不用下载）
#test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=True)  # dataset目录和此文件所在目录平级
#
import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 这里构建数据集，将两组各三个数据看成一个mini-batch，用两个3*1的矩阵表示
y_data = torch.tensor([[0.0], [0.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):           # 最少两个函数 forward前馈计算,backward是自动实现的
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()   # 直接写
        self.linear = torch.nn.Linear(1, 1) # 这里面有参数，需要写在init里面，logistic函数无参数，不用训练不用写在init里面

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))               # 可调用对象
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)  # 要不要取均值影响学习率大小取值，取均值loss会小，计算梯度是也会带有1/n，梯度会小，学习率就应大一些
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化model.parameters()即model里面所有参数，即model的linear的w和b两个参数
                                                          # lr是learning rate学习率，支持不同部分不同学习率

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())  # loss是张量

    optimizer.zero_grad()  # 记得归零，否则会累计
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())  # w是张量
print('b = ', model.linear.bias.item())  # b是张量

x_test = torch.tensor([[2.5]])
y_test = model(x_test)
print('y_pred = ', y_test.data)






