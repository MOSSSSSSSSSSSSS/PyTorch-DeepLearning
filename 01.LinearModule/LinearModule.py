import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):  # forward函数功能
    return x * w

def loss(x, y):  # loss函数功能
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_list = []  # 列表
mse_list = []  # loss用mean square error 平均平方误差，用来绘图横坐标w，纵坐标loss
for w in np.arange(0.0, 4.1, 0.1):  # numpy用法
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):  # zip()用法
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
    w_list.append(w)  # 列表append()
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)  # matplotlib画图
plt.ylabel('loss')
plt.xlabel('w')
plt.show()