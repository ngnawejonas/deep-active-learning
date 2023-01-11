# -*- coding: utf-8 -*-
"""cifar10 autoattack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hARkBQpRwNh-iFNBJKLOqKNQEcKk8Qgm
"""

# !pip install git+https://github.com/fra31/auto-attack

# https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
# Load in relevant libraries, and alias where appropriate
# from autoattack import AutoAttack 
from tqdm import tqdm
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd


import random
def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# Define relevant variables for the ML task
batch_size = 128
num_classes = 10
learning_rate = 0.1
num_epochs = 200
optparams = {'weight_decay': 0.0005, 'momentum': 0.9}
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@title cifar10_handler
class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

# Data.py
def get_CIFAR10(pool_size):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    dtrain =  CIFAR10_Handler(data_train.data[:pool_size], torch.LongTensor(data_train.targets)[:pool_size])
    dtest = CIFAR10_Handler(data_test.data[:pool_size], torch.LongTensor(data_test.targets)[:pool_size])
    return dtrain, dtest

train_dataset, test_dataset = get_CIFAR10(45000)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

#@title resnet model
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
# src: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def embedding(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out

    def forward(self, x):
        self.e1 = self.embedding(x)
        out = self.e1.view(self.e1.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

model = ResNet18().to(device)

#Setting the loss function
cost = nn.CrossEntropyLoss()

#Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **optparams)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#
#this is defined to print how many steps are remaining when training
total_step = len(train_loader)

# for i, (x,y, idx) in enumerate(train_loader):  
#   print(i, x.shape, y.shape, len(idx))
#   break

total_step = len(train_loader)
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels, idx) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        	
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        		
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    scheduler.step()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, idx in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# epsilon = 0.3
# adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in tqdm(test_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         x_adv = adversary.run_standard_evaluation(images, labels)
#         outputs = model(x_adv)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



# !pip install cleverhans


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 8/255., 'eps_iter': 0.005, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 128,
                                           shuffle = True)
for images, labels, idx in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('AAccuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# # import matplotlib.pyplot as plt
# def show(im, predicted, label, txt):
# # im1 = images[0][0].cpu().numpy()
#   plt.figure()
#   im = im[0].detach().cpu().numpy()
#   plt.imshow(im)
#   # print(predicted.cpu().numpy(), label.cpu().numpy(), (label==predicted).cpu().numpy())
#   title = txt+' {} {} {}'.format(predicted.cpu().numpy(), label.cpu().numpy(), (label==predicted).cpu().numpy())
#   plt.title(title)

# from re import X
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
set_seeds(42)
pgd_params = {'eps': 0.3, 'eps_iter': 0.1, 'nb_iter': 20, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
xcorrect = 0
total = 0
distances = []
xtest_dataset = torch.utils.data.Subset(test_dataset, np.arange(10))
test_loader = torch.utils.data.DataLoader(dataset = xtest_dataset,
                                           batch_size = 1,
                                           shuffle = False)  
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, **pgd_params)
    outputs = model(images)
    xoutputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    _, xpredicted = torch.max(xoutputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    xcorrect += (xpredicted == labels).sum().item()
    # print()
    # show(images, predicted, labels, 'clean')
    # show(x_adv, xpredicted, labels, 'adv')
    # print('---')
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
print('AAccuracy of the network on the 10000 test images: {} %'.format(100 * xcorrect / total))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 0.3, 'eps_iter': 0.05, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
pgd_params = {'eps': 0.3, 'eps_iter': 0.05, 'nb_iter': 100, 'norm': np.inf, 'targeted': False, 'rand_init': True}

correct = 0
total = 0
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    x_adv = pgd(model, images, y=labels, **pgd_params)
    outputs = model(x_adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# pgd_params = {'eps': 0.3, 'eps_iter': 0.1, 'nb_iter': 50, 'norm': np.inf, 'targeted': False, 'rand_init': True}
# 
# correct = 0
# total = 0
# 
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
#                                            batch_size = 1,
#                                            shuffle = True)
# for images, labels in tqdm(test_loader):
#     if total > 300:
#       break
#     images, labels = images.to(device), labels.to(device)
#     x_adv = pgd(model, images, y=labels, **pgd_params)
#     outputs = model(x_adv)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
#     
# print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

np.zeros(len(images))

import numpy as np
dis_list = [np.inf, 1e-2, 0.3, np.inf, 3]
valid_dis_list = np.array([x for x in dis_list if x!=np.inf])

valid_dis_list

dis_list = [np.inf, np.inf, np.inf]
valid_dis_list = np.array([x for x in dis_list if x!=np.inf])

len(valid_dis_list)