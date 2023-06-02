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
from torch.nn import functional as F
import cv2
# Recognize a single character

# 图片大小
TARGET_IMAGE_SIZE = (32, 32)

#the array of license plate character
match = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
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
            nn.Conv2d(1, 6, 5),  # in_channels=1, out_channels=6
            nn.ReLU(),  # Use ReLU activation function
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),  # in_channels=6, out_channels=16
            nn.ReLU(),  # Use ReLU activation function
            nn.MaxPool2d(2, 2)
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),  # Input size is now 16*5*5
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, len(match))
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))  # Flatten the tensor
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

def run_recognition(opt):
    net = LeNet()
    print(net)
    # sets the learning rate and number of epochs to use during training.
    lr, num_epochs = 0.005, 300
    batch_size = 256
    # creates an optimizer object using the Adam optimization algorithm,
    # which is used to update the weights of the network during training.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    # sets the file path for the saved model checkpoint.
    checkpoint_save_path = "./LeNet70.pth"
    if os.path.exists(checkpoint_save_path):
        print('load the model')
        # loads the saved checkpoint if it exists.
        net.load_state_dict(torch.load(checkpoint_save_path))
    else:
        print("不存在相应的权重文件，程序结束")
        sys.exit()

    pre_transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the data to 0-1
    ])
    preset = ImageFolder(opt.source, transform=pre_transform)
    pre_iter = DataLoader(preset)
    ans = predict(pre_iter, net, device)
    print('the result is: ' + ans)
    print('===============================The end of recognition procedure===============================')
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./singledigit/IMG_0180_0', help='picture file')
    opt = parser.parse_args()
    run_recognition(opt)
    # pre_path = './singledigit/IMG_0154_0/'
    sys.exit()

