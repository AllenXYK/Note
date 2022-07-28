## 引言
### 深度学习
深度学习的分支，以人工神经网络为基础，对数据的特征进行学习的方法

### 机器学习和深度学习的区别
特征抽取：机器学习是人工的特征抽取过程，深度学习是自动的进行特征抽取
数据量：机器学习数据少，深度学习数据多

### 常见的深度学习框架
Tensorflow Caffe2 Keras Pytorch DyNet 等

## 神经网络

### 概念
模拟生物的神经系统，对函数进行估计或近似

### 神经元
神经网络中的基础单元，相互连接组成神经网络的,一个神经元的功能是求得输入向量与权向量内积后，经过一个非线性传递函数得到一个标量的结果。
t=f（w^TA+B)

### 感知机
两层的神经网络，一个输出；简单的二分类模型，给定阈值，判断数据属于哪一部分

### 激活函数
![激活函数](https://github.com/AllenXYK/Note/blob/main/NLP/171542.png)
#### 举例
sigmoid：（0，1)
tanh:(-1,1)
relu:max(0,x)
ELU:a(e^x-1)
#### 作用
- 增加模型的非线性分割能力
- 提供模型的稳健型（鲁棒性）robust
- 缓解梯度消失的问题
- 加速模型收敛
## Pytorch
###创建Tensor
- 通过Python数组创造
- 通过numpy的数组创造
- 通过torch的api创造
torch.randint(low=,high,size=[])
torch.randn([]) 创建三行四列的随机数的tensor，随机值的分布式均值为0，方差为1

### 张量的方法和属性
- 数据中只有一个值 tensor.item
- 转化成numpy数组的时候tensor.numpy
- 获取形状
	- tensor.size()
	- tensor.size(1)获取第一个维度
- 形状的变化tensor.view([shape])
- 获取阶数tensor.dim()
- 常用计算方法tensor.max/min/std
- 转置
	-二维tensor.t() 、tensor.transpose(0,1)
	-高维tensor.transpose(1,2) 、tensor.permute(0,2,1)
- 取值和切片
t[1,:,:]
### 数据类型
- 指定数据类型torch.tensor(array，dtype)
- 获取数据类型torch.type
- 修改数据类型torch.float\long\int
tensor默认单浮点型int，经常会使用到long

###GPU的tensor的使用
- cuda.is_available()
- torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
- 在GPU上计算的结果也为cuda的数据类型,predict.cpu().detach().numpy()
## 梯度下降
### 常见的导数计算
链式法则


## Pytorch完成线性回归
### 手动线性回归
- tensor(data,required_grad=True)该tensor后续会被计算梯度
- tensor所有的操作都会被记录在grad_fn
- with torch.no_grad():其中的操作不会被追踪
- output.backward() 反向传播
- x.grad 累加梯度，每次反向传播前需要先把梯度置为0之后
- tensor.data获取tensor中的数据
- tensor.detach().numpy能够实现对tensor中数据的深拷贝，转化成ndarray的格式

### 父类子类与继承实例
- 父类做出基础功能的实现，通过子类去扩展父类的功能
- _init_调用父类/基类、子类构造函数
- parentmethod 调用父类方法
- parent.parentattr=attr 父类属性，初始化数据更新
````
class Parent:
    parentAttr = 100
    def __int__(self):
        print('调用父类构造函数')
    def parentMethod(self):
        print('调用父类方法')
    def getAttr(self,attr):
        Parent.parentAttr = attr
    def push(self):
        print(Parent.parentAttr)

class Child(Parent):
    def __int__(self):
        print('调用子类构造方法')
    def childMethod(self):
        print("调用子类方法")

c = Child()
c.childMethod()
c.parentMethod()
c.push()
c.getAttr(400)
c.push()

````
- 调用子类方法
- 调用父类方法
- 100
- 400

#### 实用小工具
issubclass(Child,Parent)------------True

isinstance(c,Child)-----------------True

#### self究竟是什么？
[Python面向对象的self究竟是什么？](https://www.bilibili.com/video/BV1rB4y1u7mT/?spm_id_from=333.788.recommend_more_video.-1&vd_source=ba07baf3007c2c07c0b618412bc3d513)
self就是一个实例对象
````
#通过实例对象和类分别调用函数的时候区别明显
dog = Animal()
dog.play()
Animal.paly(dog)
````
### 自动线性回归
#### nn.Moudle
- _init_需要调用super方法，继承父类的属性和方法
- forward方法必须实现，用来定义我们网络前向计算的过程
````
from torch import nn
class Lr(nn.Moudle):
	def _init_(self):
		super(Lr,self)._init_()#继承父类init的参数
		self.linear = nn.Linear（1，1）
	def forward（self，x）：
		out = self.linear（x）
		return out
````
- 这里nn.Linear是torch预定好的线性模型（全连接层）
- 定义了_call_方法，实现就是调用forward方法，实例才可以传入传出参数

#### 优化器类
可以理解为torch为我们封装好的用来更新参数的方法

torch.optim.SGD(参数，学习率)

torch.optim.Adam(参数，学习率)

实例如下：
````
optimizer = optim.SGD(model.parameters(),lr = 1e-3)#实例化
optimizer = optim.zero_()#梯度置为零
loss.backward()#计算梯度
optimizer.step()#更新参数的值
````

#### 损失模块
1.均方误差 nn.MSELoss()常用语分类问题

2.交叉熵损失 nn.CrossEntropyLoss（）常用语逻辑回归

````
model = Lr() #实例化模型
criterion = nn.MSELoss()# 实例化损失函数
optimizer = optim.SGD(model.parameters(),lr = 1e-3) #实例化优化器类
for i in range（100）：
	y_predict = model(x_true)#向前计算预测值
	loss = criterion(y_true,y_predict)#调用损失函数，得到损失结果
	optimizer.zero_grad()#梯度置为0
	loss.backward()#计算梯度
	optimizer.step()#更新参数的值
	
````
## 常见的最优化算法介绍
1.梯度下降：全局最优(有可能局部最优)

2.随机梯度下降：随机的从样本中抽取一个样本梯度的跟新
    api torch.optim.SGD()

3.小批量梯度下降：抽取多组数据计算梯度，防止噪声过多

<font color=red> 4.动量法：对梯度进行平滑处理，防止振幅过大</font>

5.AdaGrad：自适应学习率

6.RMSProp：让步长越来越小

7.Adam：梯度振幅不会过大

## 数据加载
### 数据集类
1.torch.utils.data.Dataset
2.完成__getitem__获取索引位置的一条数据
3.完成__len__获取数据的总个数
### 数据加载器类
1.torch，utils.data.Dataset
2.DataLoader（dataset = dataset，batch_size = 
- dataset:提前定义dataset的实例
- batch_size:出入数据的batch的大小，常用128，256等等
- shuffle：bool类型，表示是否在每次获取数据的时候提前打乱数据
- num_workers:加载数据的线程数
### pytorch自带数据集
pytorch中自带的数据集由两个上层api提供，分别是、
1.torchvision
图像
2.torchtext
文本
#### 数据下载
直接对torchvision.datasets中的数据集类都是继承自Dataset
MINIST APi 中的参数要注意：

torchvision.datasets.MNIST(root'/files/',train = True,download = True,transform = )

#### MNist数据的处理api
- torchvision.transform.ToTensor()
    - 把image对象或者（h，w，c）转化成（c，h，w）
- torchvision.transform.Normalize(mean,std)
    - 均值和标准差的形状和通道数相同
- torchvision.transforms.Compose(transforms)
    - 传入list
    - 数据经过list中的每一个方法挨个进行处理
#### 模型的构建
- 激活函数的使用
    - import torch.nn.functional as F
    - x = F.relu(x)
- 每一层数据的形状
- 交叉熵损失
    - loss = -\sum Y_true*log(P)
    - P是softmax的概率
    - 带权损失：loss = -\sum w_i x_i
- 熵：我们希望在评估模型的时候能有货币体系一样的评价标准
- 熵：描述一个系统的混乱程度
- [交叉熵](https://www.bilibili.com/video/BV15V411W7VB/?vd_source=ba07baf3007c2c07c0b618412bc3d513)
#### 模型的保存和加载：
- 保存：torch.save(model.state_dict(),path)
- 保存：torch.save(optimizer.state_dict(),path)
- 加载：model.load_state_dict(torch.load("./model.pkl"))
- 加载：optimizer.load_state_dict(torch.load('./optimizer.pkl'))

#### 模型的评估
- 首先取消对计算进行追踪 with torch.no_grad():
- 损失
- 准确率
    - 获取预测值：tensor.max(dim = -1)[-1]
    - tensor.eq(tensor2).float().mean()

## RNN(一)
tokenization(分词）:每个词语是一个token
分词方法：
- 转化为单个字
- 切分词语
N-gram：
- 准备词语特征的方法
### 文本的量化
1.one-hot 使用稀疏向量表示，占用空间多
2.word.embeding：
    a.浮点型的精密矩阵来表示tokrn
    b.向量中的每一个值都是一个超参数，随机生成并且可以进行训练
    c.api：torch.nn.Embedding(词典数量，embedding的维度)
    d.形状的变化：[batch_size,seq_len]--->[batch_size,seq_len,embedding_dim]
### <font color='red'>文本情感分类
##### 1.数据集
当dataset中的返回input结果是字符串的时候，可以通过修改collate_fn来实现
```
def my_collate(batch):
    label ,content = list(zip(*batch))
    return label,content
````
word_sequence的准备
- 定义字典保存所有词语
- 根据词频对词语进行保留
- 一个batch中对应句子的长度进行统一
- 实现方法把句子转化为序列和反向操作
###### 文本序列化
实现文本序列化需要考虑以下几点：
- 1.如何使用字典把词语和数字进行对应
- 2.不同的词语出现的次数不尽相同，是否对高频或者低频词语进行过滤，以及总的词语数量是否进行限制
- 3.得到词典之后，如何把句子转化为数字序列，如何把数字序列转化为句子
- 4.不同句子长度不相同，每个batch的句子如何构造成相同长度（可以对短句子进行填充，填充特殊字符）
- 5.对于新出现的词语在词典中没有出现怎么办（可以用特殊数字代理）
##### 思路分析
- 1.对所有句子进行分词
- 2.词语存入字典，根据次数对词语进行过滤，并统计次数
- 3.实现文本转数字序列的方法
- 4.实现数字序列转文本方法
#### 2.模型
- embedding的使用，实例化（词语的总数，embedding_dim)

#### 3.训练
#### 4.评估</font>
### zip是一个对象，可以用list强制转换！！！！
## 循环神经网络
1.概念和作用
- 时间步：time step 每个输入是在不同的时间步上的
- 循环：下一个时间步上 (链式法则)
- 具有短期记忆的网络结构，把之前的输入作为下一个时间步的输入

2.RNN类型：
- one-one 图像分类
- one-many 图像转文字
- many-one 文本分类
- many-many
    - 同步：视频分类
    - 异步：文本翻译 
  
3.LSTM 
- long short-term memory :LSTM
- C保存的就是记忆
- 其修改可以通过sigmoid和t-1点成实现
- 遗忘门：通过sigmoid决定哪些信息被遗忘
- 输入门：
    - sigmoid：决定那些信息被遗忘
    - tanh决定输入什么信息
- 输出门：
    - hidden_state
    - h_t
- 输出和输入：
    - hidden_state X_t C_t-1
    - H_t C_t 

4.GRU
- LSTM的变形
- hidden_state 
- x_t hidden_state_t-2

5.双向LSTM\GRU


## 梯度消失和梯度爆炸
1.梯度消失：梯度太小，无法进行参数的跟新，梯度小到数据类型无法表现出现NaN

2.梯度爆炸：梯度太大，大数据类型无法表示出现NaN

3.解决方法：

    - 使用更加容易计算梯度的激活函数
    - 使用改进的优化算法
    - 使用batch_Normalize

 
