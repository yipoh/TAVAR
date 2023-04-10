from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.data
import pandas as pd
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import cv2
import torchvision.models as models
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from prefetch_generator import BackgroundGenerator


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

def mos_resnet_50( **kwargs):
    model = ResNet2(Bottleneck, [3, 4, 6, 3],  ** kwargs)
    return model

class ResNet50_sce(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet50_sce, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_prob = 0.5
        self.fc1_1 = nn.Linear(512 * block.expansion, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 6)
        self.relu3_1 = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        return out_a, f_out_a

    def forward(self, x):
        return self._forward_impl(x)

def sce_resnet_50( **kwargs):
    model = ResNet50_sce(Bottleneck, [3, 4, 6, 3],  ** kwargs)
    return model

class QS_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(QS_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_prob = 0.5
        self.fc1_1 = nn.Linear(512 * block.expansion, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

    def forward(self, x):
        return self._forward_impl(x)

def QS_resnet50( **kwargs):
    model = QS_ResNet(Bottleneck, [3, 4, 6, 3],  ** kwargs)
    return model

# --Attribute Branches-- #
class attr1_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr1_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class attr2_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr2_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class attr3_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr3_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class attr4_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr4_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class attr5_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr5_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class attr6_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr6_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

# ---GCN--- #
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum==0]=0.0000001
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.00000001
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*8)
        self.gc2_2 = GraphConvolution(nhid*8, nhid * 16)
        self.dropout = dropout
        self.fc1 = nn.Linear(nhid *16*7, 1)

    def forward(self, x, x2, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x[:, 6, :] = x2[:, 6, :] # Change center node
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2_2(x, adj))
        x = x.view(x.shape[0],1,-1)
        if x.ndim==2:
            x = x.view(1,-1)
        x = F.relu(self.fc1(x))
        x = x.reshape(( x.shape[0], -1))

        return x

# --Fusion-- #
class convNet(nn.Module):
    #constructor
    def __init__(self, resnet, Attr1_Model, Attr2_Model, Attr3_Model, Attr4_Model, Attr5_Model, Attr6_Model ):
        super(convNet, self).__init__()
        self.resnet=resnet
        self.Attr1_Model=Attr1_Model
        self.Attr2_Model=Attr2_Model
        self.Attr3_Model=Attr3_Model
        self.Attr4_Model=Attr4_Model
        self.Attr5_Model=Attr5_Model
        self.Attr6_Model = Attr6_Model

    def forward(self, x_img):
        x=self.resnet(x_img)
        x1, f1 = self.Attr1_Model(x) # Interesting Content
        x2, f2 = self.Attr2_Model(x) # Object Emphasis
        x3, f3 = self.Attr3_Model(x) # Vivid Color
        x4, f4 = self.Attr4_Model(x) # Depth of Field
        x5, f5 = self.Attr5_Model(x) # Color Harmony
        x6, f6 = self.Attr6_Model(x) # Good Lighting
        return f1, f2, f3, f4, f5, f6

class convNet_GCN(nn.Module):
    def __init__(self, attr_net, scene_net,score_net, gcn_net):
        super(convNet_GCN, self).__init__()
        self.AttrNet=attr_net
        self.ScenceNet = scene_net
        self.score_net = score_net
        self.GCNNet = gcn_net

    def forward(self, x_img):

        x0, Scence_f=self.ScenceNet(x_img)
        score_f = self.score_net(x_img)
        Content_f, Object_f, VividColor_f, DoF_f, ColorHarmony_f, Light_f = self.AttrNet(x_img)

        temp1 = torch.zeros(Scence_f.shape[0], 7, Scence_f.shape[1])
        temp2 = torch.zeros(Scence_f.shape[0], 7, Scence_f.shape[1])
        for num in range(Scence_f.shape[0]):
            temp1[num : :] = torch.stack((Content_f[num,:], Object_f[num,:], VividColor_f[num,:], DoF_f[num,:], ColorHarmony_f[num,:], Light_f[num,:], Scence_f[num,:]), 0)
            temp2[num::] = torch.stack((Content_f[num, :], Object_f[num, :], VividColor_f[num, :], DoF_f[num, :],
                                        ColorHarmony_f[num, :], Light_f[num, :], score_f[num, :]), 0)
        edges_unordered = np.genfromtxt("cora.cites", dtype=np.int32)  # 读入边的信息
        adj = np.zeros((7, 7))
        for [q, p] in edges_unordered:
            adj[q - 1, p - 1] = 1
        adj = torch.from_numpy(adj)
        adj = normalize(adj)
        adj = torch.from_numpy(adj)
        adj = adj.clone().float()
        adj = adj.to(device)
        temp1=temp1.to(device)
        temp2 = temp2.to(device)
        out_a= self.GCNNet(temp1,temp2, adj)
        return out_a

# --Data-- #
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, impath ,test_name, order):
        self.impath = impath
        self.test_name = test_name
        self.YTest_order = torch.FloatTensor(order)

    def __getitem__(self, index):
        im = cv2.cvtColor(cv2.imread(self.impath + '\\' + self.test_name[index], 1), cv2.COLOR_BGR2RGB)
        im=cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        return torch.from_numpy(im), self.YTest_order[index]
    def __len__(self):
        return (self.test_name).shape[0]

def mytest(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data[:, :, 10:10 + 224, 10:10 + 224]
            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            out1= model(data)
            Score = out1[:, 0].detach().cpu().numpy()
            print()
            print(batch_idx)
            print('Test IAA score:', Score[0])


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")

    test_all_lbs = pd.read_csv('./TestList.csv')  # 读取csv文件
    impath = './TestSet'
    test_name_list = np.array(test_all_lbs['imageName'])  # 创建存放image name 的数组
    test_inds = np.argsort(test_name_list)  # 被选中的测试数据，可通过索引部分测试
    test_name = np.array(test_all_lbs['imageName'])[test_inds]
    YTest_order = np.array(test_all_lbs['number'])[test_inds]
    test_dataset = Mydataset(impath, test_name, YTest_order)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


# ------------Visual Attribute Analysis Network------------- #
    model_ft = mos_resnet_50().to(device)
    num_ftrs = 2048
    Attr1_Model = attr1_Model(0.5, num_ftrs)
    Attr2_Model = attr2_Model(0.5, num_ftrs)
    Attr3_Model = attr3_Model(0.5, num_ftrs)
    Attr4_Model = attr4_Model(0.5, num_ftrs)
    Attr5_Model = attr5_Model(0.5, num_ftrs)
    Attr6_Model = attr6_Model(0.5, num_ftrs)
    Attr_model = convNet(resnet=model_ft, Attr1_Model=Attr1_Model, Attr2_Model=Attr2_Model, Attr3_Model=Attr3_Model, Attr4_Model=Attr4_Model, Attr5_Model=Attr5_Model, Attr6_Model=Attr6_Model )


# ------------Theme Understanding Network------------- #
    scence_model = sce_resnet_50().to(device)

# ------------Aesthetics Network------------- #
    pretrained_dict = models.resnet50(pretrained=True)
    pretrained_dict.fc=nn.Linear(2048, 256)
    QS_model = pretrained_dict.to(device)

 # ------------Bilevel Reasoning------------- #
    model_GCN = GCN(nfeat=256, nhid=256, dropout=0.5).to(device)

# ------------Fusion_FC------------- #
    model_all = convNet_GCN(attr_net=Attr_model, scene_net=scence_model, score_net=QS_model, gcn_net=model_GCN)
    model_all = model_all.to(device)

    save_nm = './TAVAR_weight.pt'
    model_all.load_state_dict(torch.load(save_nm))

    mytest(model_all, test_loader, device)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    main()

