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

        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class low_path_2x(nn.Module):
    def __init__(self):
        super(low_path_2x, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(64, 128, 3, 1, 1, 1)
        self.conv3 = ConvBNReLU(128, 256, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class low_path_4x(nn.Module):
    def __init__(self):
        super(low_path_4x, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(64, 128, 3, 1, 1, 1)
        self.conv3 = ConvBNReLU(128, 256, 3, 1, 1, 1)
        self.conv4 = ConvBNReLU(256, 512, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class low_path_8x(nn.Module):
    def __init__(self):
        super(low_path_8x, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(64, 128, 3, 1, 1, 1)
        self.conv3 = ConvBNReLU(128, 256, 3, 1, 1, 1)
        self.conv4 = ConvBNReLU(256, 512, 3, 1, 1, 1)
        self.conv5 = ConvBNReLU(512, 1024, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Origin_Res(nn.Module):
    def __init__(self):
        super(Origin_Res, self).__init__()
        self.resnet = resnet(101, 16)
        self.low_2x = low_path_2x()
        self.low_4x = low_path_4x()
        self.low_8x = low_path_8x()
        self.conv_8x = ConvBNReLU(3072, 256, 3, 1, 1, 1)
        self.conv_4x = ConvBNReLU(1024, 256, 3, 1, 1, 1)
        self.conv_2x = ConvBNReLU(512, 256, 3, 1, 1, 1)
        self.conv_all = nn.Sequential(
            ConvBNReLU(768, 256, 3, 1, 1, 1),
            ConvBNReLU(256, 256, 3, 1, 1, 1)
        )
        self.conv_out = nn.Conv2d(256, 19, kernel_size=1, bias=False)



    def forward(self, img):
        size = img.size()[-2:]
        img_2x = F.interpolate(img, (int(size[0]/4), int(size[1]/4)), mode='bilinear', align_corners=True)
        img_4x = F.interpolate(img, (int(size[0]/8), int(size[1]/8)), mode='bilinear', align_corners=True)
        img_8x = F.interpolate(img, (int(size[0]/16), int(size[1]/16)), mode='bilinear', align_corners=True)

        feat_2x = self.low_2x(img_2x)
        feat_4x = self.low_4x(img_4x)
        feat_8x = self.low_8x(img_8x)


        x1, x2 ,x3 ,x4 = self.resnet(img)


        x1_size = x1.size()[-2:]
        cat_8x = torch.cat((x4, feat_8x), dim=1)
        cat_8x = self.conv_8x(cat_8x)
        cat_8x = F.interpolate(cat_8x, size=x1_size, mode='bilinear', align_corners=True)
        cat_4x = torch.cat((x2, feat_4x), dim=1)
        cat_4x = self.conv_4x(cat_4x)
        cat_4x = F.interpolate(cat_4x, size=x1_size, mode='bilinear', align_corners=True)
        cat_2x = torch.cat((x1, feat_2x), dim=1)
        cat_2x = self.conv_2x(cat_2x)
        cat_2x = F.interpolate(cat_2x, size=x1_size, mode='bilinear', align_corners=True)

        cat_all = torch.cat((cat_8x, cat_4x, cat_2x), dim=1)
        cat_all = self.conv_all(cat_all)
        cat_all = self.conv_out(cat_all)

        cat_all = F.interpolate(cat_all, size=size, mode='bilinear', align_corners=True)
        return cat_all
