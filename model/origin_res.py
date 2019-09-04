import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet import resnet

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )

        self.bn = nn.SyncBatchNorm(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Origin_Res(nn.Module):
    def __init__(self):
        super(Origin_Res, self).__init__()
        self.resnet = resnet(101, 16)
        self.conv1 = ConvBNReLU(2048, 256, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(256, 19, 1, 1, 1, 1)

    def forward(self, img):
        size = img.size()[-2:]
        x = self.resnet(img)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x
