# 全连接又叫稠密网络dense，DNN，全连接层权重占比一般是最大的
# RNN循环神经网络，面向有序列顺序的问题，如天气预测，今天的天气与昨天天气有关，自然语言处理，语言的结构是有顺序的
# RNN利用权重共享的线性层，相比于全连接层减少了权重数量
# input输入数据的维度:seq，batch，input_size，hidden的维度：layer_num，batch，hidden_size，output的维度，seq，batch，hidden_size
# RNNCell需要input_size和hidden_size，RNN需要input_size和hidden_size和num_layer，RNN里也可以选择batch_first，将输入的batch放前面
# a model to learn "hello"->"ohlol"
import numpy as np
# 使用RNN
import torch

batch_size = 1
input_size = 4
hidden_size = 4
num_layer = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)  # 是向量

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layer):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layer)

    def forward(self, input):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden) # output 和 hidden，要output
        return out.view(-1, self.hidden_size)


model = Model(input_size, hidden_size, batch_size, num_layer)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.size())
    print(labels.size())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))

# RNNCell
import torch

batch_size = 1
input_size = 4
hidden_size = 4

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

model = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))

# 独热向量，参数多，稀疏，硬编码，使用embedding层来替代独热向量作为输入

import torch

batch_size = 1
num_class = 4
hidden_size = 8
num_layer = 1
seq_len = 5
embedding_size = 10

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data).view(seq_len, batch_size)  # 都要是长整型
labels = torch.LongTensor(y_data)  # 是向量

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layer)
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, inputs):
        inputs = self.embed(inputs)
        hidden = torch.zeros(num_layer, batch_size, hidden_size)
        out, _ = self.rnn(inputs, hidden) # output 和 hidden，要output
        out = self.fc(out)
        return out.view(-1, num_class)


model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
