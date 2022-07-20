import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt


# 搞点数据y=7x+9
x = torch.rand([500,1])
y_true = 7*x+8

# 原始准备
w = torch.rand([1,1],requires_grad=True)
b = torch.rand([1,1],requires_grad=True,dtype = torch.float32)
s_rate = 0.1

# 循环
for i in range(200):
    y_predict = x*w+b#torch.matmul(x,w)+b
    loss = (y_true-y_predict).pow(2).mean()

    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()

    w.data = w.data - w.grad * s_rate
    b.data = b.data - b.grad * s_rate



    if i%50 == 0 :
        print(w.item(),b.item(),loss.item())

plt.figure(figsize=(20,20))

plt.scatter(x.numpy(),y_true.numpy())#x.numpy().reshape(-1)
y_predict=x*w+b#torch.matmul(x,w)+b
plt.plot(x.numpy(),y_predict.detach().numpy())
plt.show()
