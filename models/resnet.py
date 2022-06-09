"""
ResNet with batch normalization layers.
From 'Deep Residual Learning for Image Recognition' by Kaiming et al.
This version has in the first convolutional layer a kernel size of 3 instead of 7 to deal better with CIFAR-10.
We added the option to set the normalization layer type by passing an argument when calling ResNet50.
"""
import torch.nn as nn
import torch.nn.functional as F

type = 'Batch Norm'

def Norm(planes, type, num_groups=2):
    if type == 'Batch Norm':
        return nn.BatchNorm2d(planes)
    elif type == 'Group Norm':
        return nn.GroupNorm(num_groups, planes)
    
    
class Basicblock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type="Batch Norm"):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm(planes, type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = Norm(planes, type=norm_type)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                Norm(planes, type=norm_type)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type="Batch Norm"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = Norm(planes, norm_type)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
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

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(64, norm_type)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_type=norm_type)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(norm_type="Batch Norm"):
    return ResNet(Basicblock, [2, 2, 2, 2], norm_type=norm_type)


def ResNet38(norm_type="Batch Norm"):
    return ResNet(Bottleneck, [2, 3, 5, 2], norm_type=norm_type)


def ResNet50(norm_type="Batch Norm"):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_type=norm_type)
