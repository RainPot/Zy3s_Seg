import torch
import torch.nn as nn
from torchstat import stat
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class PAmodule(nn.Module):
    def __init__(self, classes = 19):
        super(PAmodule, self).__init__()
        # self.x1_down_conv = ConvBNReLU(256, 512, kernel_size=1, stride=1, padding=0)
        # self.x2_down_conv = ConvBNReLU(512, 512, kernel_size=1, stride=1, padding=0)
        # self.x3_down_conv = ConvBNReLU(1024, 512, kernel_size=1, stride=1, padding=0)
        # self.x4_down_conv = ConvBNReLU(2048, 1536, kernel_size=1, stride=1, padding=0)
        # self.x0_up_conv = ConvBNReLU(128, 512, kernel_size=1, stride=1, padding=0)

        self.x4_guidence_down = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0)
        self.x4_guidence_out = nn.Conv2d(256, classes, kernel_size=1, bias=False)
        self.x4_guidence_up = ConvBNReLU(classes, 256, kernel_size=1, stride=1, padding=0)



        self.dilation18 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.dilation12 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.dilation6 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dilation1 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=1, dilation=1)

        self.x24_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)
        self.x14_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)
        self.x04_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)

        self.x014_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)
        self.x124_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)


        self.x014conv = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.x124conv = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.x234conv = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=24, dilation=24)

        self.x0124conv = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=15, dilation=15)
        self.x1234conv = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=21, dilation=21)

        self.x0124_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)
        # self.x1234_gatemap = ConvBNReLU(256, 1, kernel_size=1, stride=1, padding=0)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.GAPConv = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        self.feature_fusion = ConvBNReLU(256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x4):  #x1:256  x2:512  x3:1024  x4:2048  x0:128
        H, W = x4.size()[2:]
        # x1_down = self.x1_down_conv(self.maxpool(self.maxpool(x1)))
        # x2_down = self.x2_down_conv(self.maxpool(x2))
        # x3_down = self.x3_down_conv(x3)
        # x4_down = self.x4_down_conv(x4)
        # x0_up = self.x0_up_conv(self.maxpool(self.maxpool(self.maxpool(x0))))


        x4x0 = self.dilation1(x4)
        # x4x0 = self.dilation1(torch.cat((x4_down, x0_up), dim=1))
        x4x1 = self.dilation6(x4)
        # x4x1 = self.dilation6(torch.cat((x4_down, x1_down), dim=1))
        x4x2 = self.dilation12(x4)
        # x4x2 = self.dilation12(torch.cat((x4_down, x2_down), dim=1))
        x4x3 = self.dilation18(x4)
        # x4x3 = self.dilation18(torch.cat((x4_down, x3_down), dim=1))
        x4GAP = self.GAPConv(self.avg(x4))
        x4GAP = F.interpolate(x4GAP, size=(H, W), mode='bilinear', align_corners=True)

        x4x2_gate = self.sigmoid(self.x24_gatemap(x4x2))
        x4x2_gate_reverse = torch.ones(size=x4x2_gate.size()) - x4x2_gate
        x4x2 = x4x2 + x4x2 * x4x2_gate + x4x3 * x4x2_gate_reverse

        x4x1_gate = self.sigmoid(self.x14_gatemap(x4x1))
        x4x1_gate_reverse = torch.ones(size=x4x1_gate.size()) - x4x1_gate
        x4x1 = x4x1 + x4x1 * x4x1_gate + x4x2 * x4x1_gate_reverse

        x4x0_gate = self.sigmoid(self.x04_gatemap(x4x0))
        x4x0_gate_reverse = torch.ones(size=x4x0_gate.size()) - x4x0_gate
        x4x0 = x4x0 + x4x0 * x4x0_gate + x4x1 * x4x0_gate_reverse


        x014 = self.x014conv(torch.cat((x4x0, x4x1), dim=1))
        x124 = self.x124conv(torch.cat((x4x1, x4x2), dim=1))
        x234 = self.x234conv(torch.cat((x4x2, x4x3), dim=1))

        x124_gate = self.sigmoid(self.x124_gatemap(x124))
        x124_gate_reverse = torch.ones(size=x124_gate.size()) - x124_gate
        x124 = x124 + x124 * x124_gate + x234 * x124_gate_reverse

        x014_gate = self.sigmoid(self.x014_gatemap(x014))
        x014_gate_reverse = torch.ones(size=x014_gate.size()) - x014_gate
        x014 = x014 + x014 * x014_gate + x124 * x014_gate_reverse

        x0124 = self.x0124conv(torch.cat((x014, x124), dim=1))
        x1234 = self.x1234conv(torch.cat((x124, x234), dim=1))


        x0124_gate = self.sigmoid(self.x0124_gatemap(x0124))
        x0124_gate_reverse = torch.ones(size=x0124_gate.size()) - x0124_gate
        x0124 = x0124 + x0124 * x0124_gate + x1234 * x0124_gate_reverse

        x4_guidence = self.x4_guidence_down(x4)
        x4_guidence_out = self.x4_guidence_out(x4_guidence)
        x4_guidence = self.x4_guidence_up(x4_guidence_out)


        feat = torch.cat((x0124, x1234, x4_guidence, x4GAP), dim=1)
        out_feat = self.feature_fusion(feat)

        return out_feat, x4_guidence_out

class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, kernel_size=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, kernel_size=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, kernel_size=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, kernel_size=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, kernel_size=1)
            self.conv_out = ConvBNReLU(out_chan*5, out_chan, kernel_size=1)
        else:
            self.conv_out = ConvBNReLU(out_chan*4, out_chan, kernel_size=1)

        # self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(2048, 256, 64, norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels + 5 * 64, nclass, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x



model = PAmodule(19)
# model = ASPP()
# model = _DenseASPPHead(2048, 19)
stat(model, (2048, 54, 54))