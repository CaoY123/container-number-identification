from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn, optim
import numpy as np
import torch
import time
import sys
import cv2
import os
# 训练识别单个字符
# 集装箱编号字符数组
match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '7', 9: '9', 10: 'A', 11: 'B', 12: 'C',
            13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: "I", 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: "O",
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# 准备一个由文件夹中的灰度图像组成的PyTorch数据集来训练机器学习模型。
data_path = './generated_images'
data_transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset=ImageFolder(data_path,transform=data_transform)

# 指定数据集的10%将用于验证。
validation_split=.1
# 指定数据集在拆分之前将被打乱。
shuffle_dataset = True
# 指定洗牌的随机种子。
random_seed= 42
# 指定每个DataLoader的批处理大小。
batch_size=100
# 获取数据集的长度。
dataset_size = len(dataset)
# 为数据集创建索引列表。
indices = list(range(dataset_size))
# 计算拆分数据集以进行验证的索引。
split = int(np.floor(validation_split * dataset_size))
# 如果shuffle_dataset为True，则打乱索引。
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
# 将指标分为训练指标和验证指标。
train_indices, val_indices = indices[split:], indices[:split]
# 创建一个抽样器，从训练指标随机抽样。
train_sampler = SubsetRandomSampler(train_indices)
# 创建一个抽样器，从验证指标随机抽样。
test_sampler = SubsetRandomSampler(val_indices)

# 为训练集创建一个DataLoader。
train_iter = DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
# 为验证集创建一个DataLoader。
test_iter = DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)


# lenet 训练
# 将父目录追加到系统路径，以便从中导入模块。
sys.path.append("..")
# 检查图形处理器是否可用，并相应地设置设备。
# 如果有GPU可用，则设备设置为CUDA，否则设置为CPU。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义一个名为LeNet的类，它继承自nn。模块，它是基类
# 用于PyTorch中的所有神经网络模块。
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 为网络的卷积层定义一个顺序容器。
        # 它由两对卷积层组成，后面是sigmoid激活函数和最大池化层。
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # 为网络的全连接层定义顺序容器。
        # 它由三个线性层组成，后面是sigmoid激活函数。
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 65)
        )

    # 定义神经网络的前向传递。
    # 它把一个图像张量作为输入，然后让它通过卷积层，
    # 将输出压扁，并将其通过全连接层以产生输出张量。
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

# 计算神经网络模型在给定数据集上的准确性。
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定设备，则使用网络设备
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模型，这将关闭 dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 回到训练模式
            else: # 自定义模型，在3.13节之后不再使用，无论GPU如何
                if('is_training' in net.__code__.co_varnames): # 如果你有is_training参数
                    # 设置 is_training False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 为方便使用，此函数已保存在包d2lzh_pytorch中
# 使用监督学习在由训练数据和测试数据组成的数据集上训练神经网络。
def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            x=np.array(X)
            Y=np.array(y)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


net = LeNet()
print(net)
# 设置在训练期间使用的学习率和epoch数。
lr, num_epochs = 0.001, 50
batch_size=256
# 使用Adam优化算法创建一个优化器对象，用于在训练期间更新网络的权重。
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# 为保存的模型检查点设置文件路径。
checkpoint_save_path = "./LeNet1.pth"
if os.path.exists(checkpoint_save_path ):
    print('load the model')
    # 加载保存的检查点(如果存在)。
    net.load_state_dict(torch.load(checkpoint_save_path))
else:
    # 使用train函数训练网络。
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    # 将训练过的模型检查点保存到指定的文件路径。
    torch.save(net.state_dict(),checkpoint_save_path)
    print('===============================The end of train procedure===============================')
    sys.exit()

