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
from PIL import Image, ImageDraw, ImageFont
# Recognize a single character

# 图片大小
TARGET_IMAGE_SIZE = (32, 32)

# 计算图像在指定轴上的投影。它接收一个二值化的图像和一个轴参数。返回在指定轴上的投影。
# 当axis=0时，计算垂直投影；当axis=1时，计算水平投影。
def projection(img, axis=0):
    return np.sum(img > 128, axis=axis)

# 找到投影中的谷值，即字符间的空隙。这个函数接受投影数据，以及最小宽度和最小深度作为参数。
# 最小宽度和最小深度用于筛选谷值，确保找到的谷值是字符之间的有效空隙。
def find_valleys(projection, min_width=5, min_depth=15):
    valleys = []
    width = 0
    start_idx = 0
    for idx, value in enumerate(projection):
        if value < min_depth:
            if width == 0:
                start_idx = idx
            width += 1
        else:
            if width >= min_width:
                valleys.append((start_idx, idx))
            width = 0

    # 添加最后一个区间
    valleys.append((start_idx, len(projection)))

    return valleys

# 根据谷值分割图像。这个函数接收一个图像和谷值列表，然后根据谷值将图像分割成多个部分。
def split_lines(image, valleys, flag=0):
    lines = []
    if len(valleys) > 1:
        for i in range(len(valleys) - 1):
            start = valleys[i][1]
            end = valleys[i + 1][0]
            if flag == 0:
                lines.append(image[:, start:end])
            else:
                lines.append(image[start:end, :])
    return lines

#the array of license plate character
match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '7', 9: '9', 10: 'A', 11: 'B', 12: 'C',
            13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: "I", 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: "O",
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# lenet training
# appends the parent directory to the system path to import modules from it.
sys.path.append("..")
# checks whether a GPU is available and sets the device accordingly.
# If a GPU is available, the device is set to CUDA, otherwise it is set to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# defines a class named LeNet that inherits from nn.Module, which is the base class
# for all neural network modules in PyTorch.
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # defines a sequential container for the convolutional layers of the network.
        # It consists of two pairs of convolutional layers followed by sigmoid activation functions and max pooling layers.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # defines a sequential container for the fully connected layers of the network.
        # It consists of three linear layers followed by sigmoid activation functions.
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 65)
        )

    # defines the forward pass of the neural network.
    # It takes an image tensor as input and passes it through the convolutional layers,
    # flattens the output, and passes it through the fully connected layers to produce the output tensor.
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

# It takes an image and a neural network model as input and returns the predicted label of the image.
def predict(img,net,device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # if don't specify the device, then use the net's device
        device = list(net.parameters())[0].device
    res=''

    with torch.no_grad():
        for X, y in img:
            if isinstance(net, torch.nn.Module):
                net.eval()  # evaluate model, this will close dropout
                X = X.to(device)  # move input tensor to the device
                net.to(device)  # move network parameters to the device
                temp = net(X).argmax(dim=1)
                temp = temp.cpu().numpy()
                for i in temp:
                    res += str(match[i])
                net.train()  # Go back to training mode
    return res

net = LeNet()
print(net)
# sets the learning rate and number of epochs to use during training.
lr, num_epochs = 0.001, 20
batch_size=256
# creates an optimizer object using the Adam optimization algorithm,
# which is used to update the weights of the network during training.
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# sets the file path for the saved model checkpoint.
checkpoint_save_path = "./LeNet1.pth"
if os.path.exists(checkpoint_save_path ):
    print('load the model')
    # loads the saved checkpoint if it exists.
    net.load_state_dict(torch.load(checkpoint_save_path))
else:
    print("不存在相应的权重文件，程序结束")
    sys.exit()


pre_path = './singledigit/IMG_0154_0/'

pre_transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.CenterCrop(size=(32,32)),
    transforms.ToTensor()
])
preset=ImageFolder(pre_path,transform=pre_transform)
pre_iter = DataLoader(preset)
ans=predict(pre_iter,net, device)
print('the result is: '+ans)
print('===============================The end of recognition procedure===============================')
sys.exit()

