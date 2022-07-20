import torch
from torch.utils.data import Dataset,DataLoader

data_path = r"C:\Users\Alen\Desktop\gitcode\Note\Pytorch_NPL\SMSS.txt"

# 完成数据集类

class MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path).readlines()
    def __getitem__(self, item):
        cur_line =  self.lines[item].strip()
        lable = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return lable,content
    def __len__(self):
        return len(self.lines)
my_dataset = MyDataset()
data_loader = DataLoader(dataset=my_dataset,batch_size=2,shuffle=True)
if __name__ == '__main__':
    # my_dataset = MyDataset()
    # for i in range(len(my_dataset)):
    #     print(i,my_dataset[i])
    for i in data_loader:
        print(i)
