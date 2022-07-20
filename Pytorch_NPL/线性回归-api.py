import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
# 定义数据

x=torch.rand([500,1])
y=3*x+8
# 定义模型
class Lr(nn.Module):
    def __init__(self):
        super(Lr,self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self,x):
        out = self.linear(x)
        return out
# 实例化模型，loss，优化模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
x,y = x.to(device),y.to(device)

model = Lr().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#训练模型

for i in range(20000):
    out = model(x)
    loss = criterion(y,out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%20 == 0:
        #params = list (model.parameters())
        #print(params[0].item,params[1].item)
        print('Epoch[{}/{}],loss:{:8f}'.format(i,20000,loss.data))

# 模型评估
model.eval()#设置模型为评估模式，即预测模式
predict = model(x)
predict =predict.cpu().data.numpy()
plt.plot(x.cpu().data.numpy(),predict)
plt.scatter(x.cpu().data.numpy(),y.cpu().data.numpy())
plt.show()