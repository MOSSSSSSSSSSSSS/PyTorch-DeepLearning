# mini-batch用到了dataloader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader  # Dataset是抽象类，只能继承不能实例化

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape是N，9，[0]就是N
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,   # 是否打乱
                          num_workers=2)  # 几个线程去读，多线程


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
    for index, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# example mnist dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_set = datasets.MNIST(root='../dataset/mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_set = datasets.MNIST(root='../dataset/mnist',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

train_loader = DataLoader(dataset=train_set,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(dataset=test_set,
                         batch_size=32,
                         shuffle=False)
for i in range(100):
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
         # ............