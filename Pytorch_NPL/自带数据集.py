from torchvision.datasets import  MNIST
minist = MNIST(root="./data",train=True,download=True)
print(minist)
print(minist[0][0].show())