# 用pytorch提供的工具实现线性模型
# 1.dataset 2.design model using class inherit from nn.module(用来计算 y hat)这里是y = W * x + b（一个线性单元）
# 3.construct loss and optimizer(优化器) 最终loss是一个标量，这里是batch中所求的各loss相加求均值。若loss是个向量是不能backward的
# 4. training cycle (forward backward update)
# numpy有广播机制
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 这里构建数据集，将两组各三个数据看成一个mini-batch，用两个3*1的矩阵表示
y_data = torch.tensor([[2.0], [4.0], [6.0]])  # 都是二维张量，也就是一个3*1矩阵，因此y = W * x + b,W会转变成矩阵和x乘，b也是3*1的矩阵和乘积结果各项相加

class LinearModel(torch.nn.Module):           # 最少两个函数 forward前馈计算 module构造的对象,backward是自动实现的
    def __init__(self):
        super(LinearModel, self).__init__()   # 直接写
        self.linear = torch.nn.Linear(1, 1)  #torch.nn.linear是一个类继承自module，这是在构造一个对象，参数in_features,out_features,bias = true
                                                                  #in/out_features是size of input/output sample，行表示样本，列表示feature
                                                                  #进行运算的时候要求转置 y的转置 = w的转置 * x的转置 + b
    def forward(self, x):
        y_pred = self.linear(x)               # 可调用对象
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)  # 继承自nn.module，参数有size_average=True（要不要求均值）,reduce=True（要不要降维）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化模块的SGD类，第一个参数是要优化model.parameters()即所有参数,lr是learning rate学习率，支持不同部分不同学习率

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)  # loss是一个对象，打印的时候自动调用__str__()，不会产生计算图，所以安全
    print(epoch, loss.item())

    optimizer.zero_grad()  # 记得归零，否则会累计
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)






