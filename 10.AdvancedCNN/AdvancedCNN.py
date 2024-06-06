# 构建更复杂的神经网络，使用类减少代码的冗余
# Inception(盗梦空间结构)是经典模型GoogLeNet中最核心的子网络结构，Google团队在随后2年里不断改进，相继提出了v1-v4和xcpetion结构
# Inception就是将多个卷积或池化操作放在一起组装成一个网络模块，设计神经网络时，以模块为单位去组装整个网络结构。
# 在GoogLeNet之前的网络中，一层一般只使用一种操作（单个卷积或池化），一层只使用一个操作的弊端是提取出来的特征往往过于单调。
# 在实际情况中，由于图像的不同尺度等原因，需要不同程度的下采样，或者说不同大小的卷积核，我们希望网络代替人手工自己决定到底使用1 × 1 3 × 3 5 × 5
# 以及是否需要max_pooling层，由网络自动去寻找适合的结构。并且节省计算提取更为丰富的特征，Inception结构正好满足了这样的需求，因而获得了更好的表现。
# 1*1卷积层来降低通道数可以有效减少总共运算的次数

# 2012年，针对ImageNet竞赛开发的AlexNet模型是一个包含8层的卷积神经网络。到了2014年，牛津大学的视觉几何组（VGG）通过叠加3x3卷积层将网络深度增加到了19层。
# 但是，层级的增加却导致训练精度的迅速下降，这种现象被称为“性能退化”问题。
# 那为什么网络会发生退化呢？
# 过拟合（overfitting）吗？（网络模型过度拟合训练数据），显然不是，现在的问题是深层网络的训练误差与测试误差都很大，而过拟合的现象是训练误差小、测试误差大。
# 梯度弥散/爆炸吗？反向传播时梯度一直大于1，经过层层回传，梯度将会呈几何倍数增长，这就是梯度爆炸现象；正向传播时如果梯度一直小于1，那么梯度会呈几何倍数下降，直到0，这就是梯度消失（弥散）现象。也不是因为这个原因，因为BN层的作用，BN层的本质是控制每层输入的数据分布，所以梯度问题已经被解决了。
# 那退化现象的原因是什么呢？
# 按常理，网络模型深度越深，模型的效果应该越来越好，但是随着堆叠一层又一层网络，效果似乎越来越差了，那什么都不做不就好了吗。
# ops！ 什么都不做！！！
# 由于非线性激活函数ReLU的存在，导致输入输出不可逆，造成了模型的信息损失，更深层次的网络使用了更多的ReLU函数，导致了更多的信息损失，这使得浅层特征随着前项传播难以得到保存，那么有没有什么办法能保留浅层网络提取的一些特征呢？
# 简单来说就是让模型在向更深层次前进的过程中，还有保留特征的能力，以至于不会发生退化现象。ResNet的核心思想就是引入一个恒等快捷连接（identity shortcut connection），直接跳过一个或者多个层。
# residual还在一定程度上缓解了梯度消失问题。然而，梯度消失并不是导致性能退化问题的根源，因为通过引入规范化层（如批量规范化）可在一定程度上解决此问题。
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
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, 1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, 1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, 5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, 1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, 3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, 3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, 1)

    def forward(self, x):
        branch1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, 3, 1, 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(88, 20, 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(1408, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 32)
        self.l5 = torch.nn.Linear(32, 10)
        self.incep1 = InceptionA(10)
        self.incep2 = InceptionA(20)
        self.residual1 = ResidualBlock(10)
        self.residual2 = ResidualBlock(20)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.residual1(x)
        x = self.incep1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.residual2(x)
        x = self.incep2(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # ############################################
# -----------------------------------loss optimizer--------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 更好的优化算法带动量的，动量梯度下降
# ----------------------------------------train test--------------------------------------------------------------------
def train(epoch):
    for batch_index, data in enumerate(train_loader, 0):
        # enumerate(train_loader, 0)会返回一个可迭代对象，其中batch_index是迭代的索引，而data是从train_loader加载的数据批次。
        inputs, target = data  # target 64*1的矩阵，表示是哪一类0到9
        inputs, target = inputs.to(device), target.to(device)
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
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, outputs = torch.max(outputs.data, dim=1)     # 行是第0个维度，列是第一个维度，沿着第一个维度，max函数返回value和index，_，是占位符，不要value
            total += target.size(0)
            correct += (outputs == target).sum().item()  # ##################
    print('Accuracy on test set: %d %%' % (100*correct/total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
