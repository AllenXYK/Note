import torch
from torch.utils.data import  DataLoader
from torchvision.transforms import Compose,Normalize,ToTensor
from torchvision.datasets import MNIST
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import os
import numpy as np
#准备数据集

def get_dataloader(train=True):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307),std=(0.3081))
    ])
    #mean和std的形状和通道数相同

    dataset = MNIST(root="./data",train=train,transform=transform_fn)
    data_loader = DataLoader(dataset,batch_size=128,shuffle=True)
    return data_loader

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)
    def forward(self,input):
        x = input.view([input.size(0),1*28*28])#1.用-1也可以，前面截断也可以
        x = self.fc1(x)

        x = F.relu(x)
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)
model = MnistModel()
optimizer = Adam(model.parameters(),lr=1e-3)
if os.path.exists('./model.pkl'):
    model.load_state_dict(torch.load("./model.pkl"))
    optimizer.load_state_dict(torch.load('./optimizer.pkl'))
def train(Epoch,SumEpoch):

    data_loader = get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if (idx+1)%10 == 0:
            print('Epoch[{}/{}],loading:{}%,loss:{:8f}'.format(Epoch,SumEpoch,idx/len(data_loader), loss.data))
        if idx%200 == 0:
            torch.save(model.state_dict(),'./model.pkl')
            torch.save(optimizer.state_dict(),'./optimizer.pkl')
def test():
    loss_list = []
    acc_list=[]
    test_dataloader = get_dataloader(train=False)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)# [batchsize,10]
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率
            pre = output.max(dim=-1)[-1]#dim=0求的是列的最大值
            cur_acc = pre.eq(target).float().mean()
            acc_list.append(cur_acc)
    print('准确率：{}   损失率：{}'.format(np.mean(acc_list),np.mean(loss_list)))

if __name__ == '__main__':
    # test()
    # j=5
    # for i in range(j):
    #     train(i,j)
    #     test()
    #     #进行全连接的操作
    for idx,(input,target) in enumerate(get_dataloader()):
        print(idx,input,target)