import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnet import resnet



class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

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
        self.x2_down_conv = ConvBNReLU(512, 512, kernel_size=1, stride=1, padding=0)
        self.x3_down_conv = ConvBNReLU(1024, 512, kernel_size=1, stride=1, padding=0)
        self.x4_down_conv = ConvBNReLU(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.x0_down_conv = ConvBNReLU(128, 512, kernel_size=1, stride=1, padding=0)

        self.dilation18 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.dilation12 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.dilation6 = ConvBNReLU(2048, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dilation1 = ConvBNReLU(1024, 256, kernel_size=3, stride=1, padding=1, dilation=1)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.GAPConv = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        self.feature_fusion = ConvBNReLU(256 * 5, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x1, x2, x3, x4 ,x0):  #x1:256  x2:512  x3:1024  x4:2048  x0:128
        H, W = x4.size()[2:]
        x1_down = self.x1_down_conv(self.maxpool(self.maxpool(x1)))
        x2_down = self.x2_down_conv(self.maxpool(x2))
        x3_down = self.x3_down_conv(x3)
        x4_down = self.x4_down_conv(x4)
        x0_down = self.x0_down_conv(self.maxpool(self.maxpool(self.maxpool(x0))))

        x0x1x4 = self.dilation18(torch.cat((x4_down, x1_down, x0_down), dim=1))
        x0x2x4 = self.dilation12(torch.cat((x4_down, x2_down, x0_down), dim=1))
        x0x3x4 = self.dilation6(torch.cat((x4_down, x3_down, x0_down), dim=1))
        x4self = self.dilation1(x4_down)
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

        self.stage0_conv = ConvBNReLU(128, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

        self.stage_conv = ConvBNReLU(stage_channel, 256, kernel_size=1, stride=1, padding=0)
        self.stage_spatial_conv = ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0)

        self.last_conv1 = ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = ConvBNReLU(512, 256, kernel_size=3, stride=1, padding=1)


        self.GAP = nn.AdaptiveAvgPool2d((3, 3))

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
        feat_spatial = feat * stagecur_spatial
        feat = feat + feat_spatial

        feat_stagecur_cat = torch.cat((feat, stagecur), dim=1)
        feat_stagecur_cat = self.fusion_conv(feat_stagecur_cat)

        stage_global_feat = self.GAP(feat_stagecur_cat)
        stage_global_feat = F.interpolate(stage_global_feat, size=(H, W), mode='bilinear', align_corners=True)
        similarity_map = F.cosine_similarity(feat_stagecur_cat, stage_global_feat, dim=1)
        similarity_map = similarity_map.unsqueeze(1)

        feat_stagecur_cat_global = feat_stagecur_cat * similarity_map
        feat_stagecur_cat = feat_stagecur_cat + feat_stagecur_cat_global

        feat_stagecur_cat = self.last_conv1(feat_stagecur_cat)
        feat_stagecur_cat = torch.cat((feat, feat_stagecur_cat), dim=1)
        feat_stagecur_cat = self.last_conv2(feat_stagecur_cat)

        for i in range(self.stagecur_stage0):
            stage0 = self.maxpool(stage0)
        stage0 = self.stage0_conv(stage0)
        stage0 = self.sigmoid(stage0)
        stage0_spatial = feat_stagecur_cat * stage0
        feat_stagecur_cat = feat_stagecur_cat + stage0_spatial


        return feat_stagecur_cat




class PANet(nn.Module):
    def __init__(self, classes):
        super(PANet, self).__init__()

        self.backbone = resnet(101, 16)
        self.PAModule = PAmodule()
        self.DIGModule1 = DIGModule(3, 0, 1024)
        self.DIGModule2 = DIGModule(2, 1, 512)
        self.DIGModule3 = DIGModule(1, 1, 256)
        # self.DIGModule4 = DIGModule(0, 1, 128)

        self.conv_out = nn.Conv2d(256, classes, kernel_size=1, bias=False)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]
        x1, x2, x3, x4, x0 = self.backbone(x)
        feat = self.PAModule(x1, x2, x3, x4, x0)
        feat = self.DIGModule1(feat, x3, x0)
        feat = self.DIGModule2(feat, x2, x0)
        feat = self.DIGModule3(feat, x1, x0)
        # feat = self.DIGModule4(feat, x0, x0)
        final = self.conv_out(feat)

        final = F.interpolate(final, (H, W), mode='bilinear', align_corners=True)

        return final



