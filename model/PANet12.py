import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnet import resnet

class lowpath(nn.Module):
    def __init__(self):
        super(lowpath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, 3, 1, 1, 1)
        self.conv2 = ConvBNReLU(64, 128, 3, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

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
    def __init__(self):
        super(PAmodule, self).__init__()
        self.x1_down_conv = ConvBNReLU(256, 512, kernel_size=1, stride=1, padding=0)
        self.x2_down_conv = ConvBNReLU(256, 512, kernel_size=1, stride=1, padding=0)
        self.x3_down_conv = ConvBNReLU(256, 512, kernel_size=1, stride=1, padding=0)
        self.x4_down_conv = ConvBNReLU(2048, 1536, kernel_size=1, stride=1, padding=0)


        self.dilation18 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.dilation12 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.dilation6 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dilation1 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=1, dilation=1)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.GAPConv = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        self.feature_fusion = ConvBNReLU(256 * 5, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x1, x2, x3, x4):  #x1:256  x2:512  x3:1024  x4:2048  x0:128
        H, W = x4.size()[2:]
        x1_down = self.x1_down_conv(self.maxpool(self.maxpool(x1)))
        x2_down = self.x2_down_conv(self.maxpool(x2))
        x3_down = self.x3_down_conv(x3)
        x4_down = self.x4_down_conv(x4)


        x0x1x4 = self.dilation18(torch.cat((x4_down, x1_down), dim=1))
        x0x2x4 = self.dilation12(torch.cat((x4_down, x2_down), dim=1))
        x0x3x4 = self.dilation6(torch.cat((x4_down, x3_down), dim=1))
        x4self = self.dilation1(x4)
        x4GAP = self.GAPConv(self.avg(x4))
        x4GAP = F.interpolate(x4GAP, size=(H, W), mode='bilinear', align_corners=True)

        feat = torch.cat((x0x1x4, x0x2x4, x0x3x4, x4self, x4GAP), dim=1)
        out_feat = self.feature_fusion(feat)

        return out_feat


class DIGModule(nn.Module):
    def __init__(self, stagecur_stage0 = 0, featstagecur = 0, stage_channel = 128):
        super(DIGModule, self).__init__()
        self.stagecur_stage0 = stagecur_stage0
        self.featstagecur = featstagecur

        self.stage0_conv = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

        self.stage_conv = ConvBNReLU(stage_channel, 256, kernel_size=1, stride=1, padding=0)
        self.stage_spatial_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0)

        self.last_conv1 = ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=1)


        self.GAP = nn.AdaptiveAvgPool2d((3, 3))
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


    def forward(self, feat, stagecur, stage0):
        H, W = stagecur.size()[2:]
        if self.featstagecur:
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=True)

        stagecur = self.stage_conv(stagecur)

        stagecur_spatial = self.sigmoid(self.stage_spatial_conv(stagecur))
        feat = feat * stagecur_spatial


        feat_stagecur_cat = torch.cat((feat, stagecur), dim=1)
        feat_stagecur_cat = self.fusion_conv(feat_stagecur_cat)

        stage_global_feat = self.GAP(feat_stagecur_cat)
        stage_global_feat = F.interpolate(stage_global_feat, size=(H, W), mode='bilinear', align_corners=True)
        similarity_map = F.cosine_similarity(feat_stagecur_cat, stage_global_feat, dim=1)
        similarity_map = similarity_map.unsqueeze(1)

        feat_stagecur_cat = feat_stagecur_cat * similarity_map


        feat_stagecur_cat = self.last_conv1(feat_stagecur_cat)
        feat_stagecur_cat = torch.cat((feat, feat_stagecur_cat), dim=1)
        feat_stagecur_cat = self.last_conv2(feat_stagecur_cat)

        for i in range(self.stagecur_stage0):
            stage0 = self.maxpool(stage0)
        stage0 = self.stage0_conv(stage0)
        stage0 = self.sigmoid(stage0)
        feat_stagecur_cat = feat_stagecur_cat * stage0


        return feat_stagecur_cat




class PANet(nn.Module):
    def __init__(self, classes):
        super(PANet, self).__init__()

        self.backbone = resnet(101, 16)
        self.x4_low_conv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.PAModule = PAmodule()
        self.DIGModule1 = DIGModule(3, 0, 1024)
        self.DIGModule2 = DIGModule(2, 1, 512)
        self.DIGModule3 = DIGModule(1, 1, 256)
        # self.DIGModule4 = DIGModule(0, 1, 128)

        self.low_2x = lowpath()
        self.low_4x = lowpath()
        self.low_8x = lowpath()

        self.conv_low1 = nn.Conv2d(384, 48, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv_low2 = nn.Conv2d(640, 48, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv_low3 = nn.Conv2d(1152, 48, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv_cat = nn.Sequential(
            ConvBNReLU(400, 256),
            ConvBNReLU(256, 256)
        )

        self.conv_out = nn.Conv2d(256, classes, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):

        x1, x2, x3, x4, x0 = self.backbone(x)

        x4_low = self.x4_low_conv(x4)

        x3_low = self.DIGModule1(x4_low, x3, x0)
        x2_low = self.DIGModule2(x3_low, x2, x0)
        x1_low = self.DIGModule3(x2_low, x1, x0)
        # feat = self.DIGModule4(feat, x0, x0)

        feat = self.PAModule(x1_low, x2_low, x3_low, x4)

        H, W = x1.size()[2:]

        low1_short = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        low1_short = self.low_2x(low1_short)
        low1 = torch.cat((x1, low1_short), dim=1)
        low1 = self.bn1(self.conv_low1(low1))

        low2_short = F.interpolate(x, (int(H / 2), int(W / 2)), mode='bilinear', align_corners=True)
        low2_short = self.low_4x(low2_short)
        low2 = torch.cat((x2, low2_short), dim=1)
        low2 = self.bn2(self.conv_low2(low2))

        low3_short = F.interpolate(x, (int(H / 4), int(W / 4)), mode='bilinear', align_corners=True)
        low3_short = self.low_8x(low3_short)
        low3 = torch.cat((x3, low3_short), dim=1)
        low3 = self.bn3(self.conv_low3(low3))

        feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
        low2 = F.interpolate(low2, (H, W), mode='bilinear', align_corners=True)
        low3 = F.interpolate(low3, (H, W), mode='bilinear', align_corners=True)
        cat = torch.cat((feat, low1, low2, low3), dim=1)
        final = self.conv_cat(cat)
        final = self.conv_out(final)

        H, W = x.size()[2:]
        final = F.interpolate(final, (H, W), mode='bilinear', align_corners=True)

        return final



