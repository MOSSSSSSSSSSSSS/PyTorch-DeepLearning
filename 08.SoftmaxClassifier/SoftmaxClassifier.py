# 多分类问题，如果是输出属于各分类的概率，也就是相当于每个分类是个二分类，这种方法并不好，更好的应该是输出一个分布，所有的大于0，且和为1
# 多分类问题，前面这些层还是用sigmoid函数，最后一层是softmax层，这样能得到想要的分布
# softmax函数介绍：见Graph1
# loss的计算：torch.nn.CrossEntroyLoss,见Graph
# mnist手写数字识别
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
# -----------------------------------dataset---------------------------------------------------------------------------
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 由一个28*28*1单通道像素值0到255的PIL图像，变成一个1*28*28（C H W），像素值归一化到0到1之间的浮点数
    transforms.Normalize((0.1307, ), (0.3081, ))  # mean和std，平均值和标准差
])
train = datasets.MNIST(root='../dataset/mnist',
                       train=True,
                       download=True,
                       transform=transform)
test = datasets.MNIST(root='../dataset/mnist',
                      train=False,
                      download=True,
                      transform=transform)
train_loader = DataLoader(train,
                          shuffle=True,
                          batch_size=batch_size)
test_loader = DataLoader(test,
                         shuffle=False,
                         batch_size=batch_size)
# -----------------------------------model---------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 32)
        self.l6 = torch.nn.Linear(32, 16)
        self.l7 = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return self.l7(x)

model = Net()
# -----------------------------------loss optimizer--------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 更好的优化算法带动量的，动量梯度下降
# ----------------------------------------train test--------------------------------------------------------------------
def train(epoch):
    for batch_index, data in enumerate(train_loader, 0):
        # enumerate(train_loader, 0)会返回一个可迭代对象，其中batch_index是迭代的索引
        inputs, target = data  # data是两个tensor组成的列表，第一个tensor四维是数据批次，第二个是个向量，是标签
        outputs = model(inputs)  # 64 * 10的矩阵
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

def test():
    total = 0
    correct = 0
    with torch.no_grad():  # 只是测试
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, outputs = torch.max(outputs.data, dim=1)     # 行是第0个维度，列是第一个维度，沿着第一个维度，max函数返回value和index，_，是占位符，不要value
            total += target.size(0)
            correct += (outputs == target).sum().item()  # ##################
    print('Accuracy on test set: %d %%' % (100*correct/total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()


