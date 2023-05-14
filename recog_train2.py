from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision import transforms
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import time
import sys
import cv2
import os
from PIL import Image
from torchvision.transforms import functional as F
# 训练识别单个字符
# 集装箱编号字符数组
match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '7', 9: '9', 10: 'A', 11: 'B', 12: 'C',
            13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: "I", 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: "O",
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

class CustomTransforms():
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor

    def __call__(self, img):
        img = self.add_noise(img)
        img = self.morphological_transform(img)
        return img

    def add_noise(self, img):
        img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, self.noise_factor * 255, img.shape)  # Note: noise is now a percentage of the full scale (255)
        img += noise
        img = np.clip(img, 0., 255.)
        return Image.fromarray(img.astype(np.uint8))

    def morphological_transform(self, img):
        img = np.array(img)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=2)
        return Image.fromarray(img)

custom_transforms = CustomTransforms()

# 准备一个由文件夹中的灰度图像组成的PyTorch数据集来训练机器学习模型。
data_path = './generated_images'
# In your data_transform
data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    CustomTransforms(),  # Apply your custom transform here
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Only one channel now
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
# 用于PyTorch中的所有神经网络模块。
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # 单通道输入，16个输出通道，5x5平方卷积核
        self.pool1 = nn.MaxPool2d(2, 2)   # 2x2的最大池化
        self.conv2 = nn.Conv2d(16, 32, 5) # 16个输入通道，32个输出通道，5x5平方卷积核
        self.pool2 = nn.MaxPool2d(2, 2)   # 2x2的最大池化
        self.fc1 = nn.Linear(32*61*61, 120)  # 全连接层，输入为32*61*61，输出120
        self.fc2 = nn.Linear(120, 84)       # 全连接层，输入120，输出84
        self.fc3 = nn.Linear(84, 36)        # 全连接层，输入84，输出36 (现在有36类)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32*61*61)  # 将二维特征图转为一维（这个维度可能需要根据实际情况调整）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
def train(net, train_iter, batch_size, optimizer, device, accumulation_steps=4):
    net.train()
    train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
    loss = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()  # 将梯度初始化为零
    for i, (X, y) in enumerate(train_iter):
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l = l / accumulation_steps  # Scale the loss by accumulation steps
        l.backward()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        if (i+1) % accumulation_steps == 0 or i+1 == len(train_iter):  # 更新模型参数
            optimizer.step()
            optimizer.zero_grad()  # 将梯度清零
        batch_count += 1
    train_loss = train_l_sum / batch_count
    train_acc = train_acc_sum / n
    return train_loss, train_acc


# 创建模型并指定设备
net = Net().to(device)
print(net)
# 设置在训练期间使用的学习率和epoch数。
lr, num_epochs = 0.0005, 500
batch_size=96
# 使用Adam优化算法创建一个优化器对象，用于在训练期间更新网络的权重。
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)  # 添加weight_decay参数
# 创建学习率调度器
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 为保存的模型检查点设置文件路径。
checkpoint_save_path = "./LeNet36.pth"
if os.path.exists(checkpoint_save_path ):
    print('load the model')
    # 加载保存的检查点(如果存在)。
    net.load_state_dict(torch.load(checkpoint_save_path))
    # 验证加载的模型的准确性
    test_acc = evaluate_accuracy(test_iter, net)
    print('Accuracy of loaded model on test set: ', test_acc)
else:
    # 使用train函数训练网络。
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间
        # 训练阶段
        train_loss, train_acc = train(net, train_iter, batch_size, optimizer, device)
        # 验证阶段
        test_acc = evaluate_accuracy(test_iter, net)
        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算训练该epoch的时间
        print('epoch %d, train loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (
        epoch + 1, train_loss, train_acc, test_acc, epoch_time))
        # 学习率调度
        scheduler.step()
        # 每50个epoch保存一次模型
        if epoch % 50 == 0:
            torch.save(net.state_dict(), checkpoint_save_path)
    print('===============================The end of train procedure===============================')
    sys.exit()

