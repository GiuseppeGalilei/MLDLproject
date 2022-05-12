"""
ResNet with batch normalization layers.
From 'Deep Residual Learning for Image Recognition' by Kaiming et al.

This version has in the first convolutional layer a kernel size of 3 instead of 7 to deal better with CIFAR-10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

type = 'Batch Norm'

def Norm(planes, type, num_groups=2):
    if type == 'Batch Norm':
        return nn.BatchNorm2d(planes)
    elif type == 'Group Norm':
        return nn.GroupNorm(num_groups, planes)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='Batch Norm'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = Norm(planes, norm_type)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = Norm(self.expansion*planes, norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                Norm(self.expansion*planes, norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='Batch Norm'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(self.in_planes, norm_type)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.linear = nn.Linear(16*block.expansion, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        extracted_features = out
        out = self.layer1(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, extracted_features


def ResNet8(norm_type):
    return ResNet(Bottleneck, [2], norm_type=norm_type)
