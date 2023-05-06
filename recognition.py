from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn, optim
import numpy as np
import torch
import sys
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
# Recognize a single character

# 图片大小
TARGET_IMAGE_SIZE = (62, 64)
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
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # 针对62x64图像，调整全连接层输入大小
        self.fc = nn.Sequential(
            nn.Linear(16 * 12 * 13, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 65)
        )

    def forward(self, img):
        feature = self.conv(img)
        # 针对62x64图像，调整展平特征映射的大小
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def predict(img, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    res = ''

    with torch.no_grad():
        for X, y in img:
            if isinstance(net, torch.nn.Module):
                net.eval()
                X = X.to(device)
                net.to(device)
                temp = net(X).argmax(dim=1)
                temp = temp.cpu().numpy()
                for i in temp:
                    res += str(match[i])
                net.train()
    return res
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

def run_recognition(opt):
    net = LeNet()
    print(net)
    # sets the learning rate and number of epochs to use during training.
    lr, num_epochs = 0.001, 20
    batch_size = 256
    # creates an optimizer object using the Adam optimization algorithm,
    # which is used to update the weights of the network during training.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # sets the file path for the saved model checkpoint.
    checkpoint_save_path = "./LeNet3.pth"
    if os.path.exists(checkpoint_save_path):
        print('load the model')
        # loads the saved checkpoint if it exists.
        net.load_state_dict(torch.load(checkpoint_save_path))
    else:
        print("不存在相应的权重文件，程序结束")
        sys.exit()

    # 修改预处理部分
    pre_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(62, 64)),
        transforms.ToTensor()
    ])
    preset = ImageFolder(opt.source, transform=pre_transform)
    pre_iter = DataLoader(preset)
    ans = predict(pre_iter, net, device)
    print('the result is: ' + ans)
    print('===============================The end of recognition procedure===============================')
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./singledigit/IMG_0155_0', help='picture file')
    opt = parser.parse_args()
    run_recognition(opt)
    # pre_path = './singledigit/IMG_0154_0/'
    sys.exit()

