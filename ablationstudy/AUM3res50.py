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
        self.x1_down_conv = ConvBNReLU(256, 512, kernel_size=1, stride=1, padding=0)
        self.x2_down_conv = ConvBNReLU(512, 512, kernel_size=1, stride=1, padding=0)
        self.x3_down_conv = ConvBNReLU(1024, 512, kernel_size=1, stride=1, padding=0)
        self.x4_down_conv = ConvBNReLU(2048, 1536, kernel_size=1, stride=1, padding=0)
        self.x0_up_conv = ConvBNReLU(128, 512, kernel_size=1, stride=1, padding=0)

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

    def forward(self, x1, x2, x3, x4, x0):  #x1:256  x2:512  x3:1024  x4:2048  x0:128
        H, W = x4.size()[2:]
        x1_down = self.x1_down_conv(self.maxpool(self.maxpool(x1)))
        x2_down = self.x2_down_conv(self.maxpool(x2))
        x3_down = self.x3_down_conv(x3)
        x4_down = self.x4_down_conv(x4)
        x0_up = self.x0_up_conv(self.maxpool(self.maxpool(self.maxpool(x0))))


        x4x0 = self.dilation1(torch.cat((x4_down, x0_up), dim=1))
        x4x1 = self.dilation6(torch.cat((x4_down, x1_down), dim=1))
        x4x2 = self.dilation12(torch.cat((x4_down, x2_down), dim=1))
        x4x3 = self.dilation18(torch.cat((x4_down, x3_down), dim=1))
        x4GAP = self.GAPConv(self.avg(x4))
        x4GAP = F.interpolate(x4GAP, size=(H, W), mode='bilinear', align_corners=True)

        x4x2_gate = self.sigmoid(self.x24_gatemap(x4x2))
        x4x2_gate_reverse = torch.ones(size=x4x2_gate.size()).cuda() - x4x2_gate
        x4x2 = x4x2 + x4x2 * x4x2_gate + x4x3 * x4x2_gate_reverse

        x4x1_gate = self.sigmoid(self.x14_gatemap(x4x1))
        x4x1_gate_reverse = torch.ones(size=x4x1_gate.size()).cuda() - x4x1_gate
        x4x1 = x4x1 + x4x1 * x4x1_gate + x4x2 * x4x1_gate_reverse

        x4x0_gate = self.sigmoid(self.x04_gatemap(x4x0))
        x4x0_gate_reverse = torch.ones(size=x4x0_gate.size()).cuda() - x4x0_gate
        x4x0 = x4x0 + x4x0 * x4x0_gate + x4x1 * x4x0_gate_reverse


        x014 = self.x014conv(torch.cat((x4x0, x4x1), dim=1))
        x124 = self.x124conv(torch.cat((x4x1, x4x2), dim=1))
        x234 = self.x234conv(torch.cat((x4x2, x4x3), dim=1))

        x124_gate = self.sigmoid(self.x124_gatemap(x124))
        x124_gate_reverse = torch.ones(size=x124_gate.size()).cuda() - x124_gate
        x124 = x124 + x124 * x124_gate + x234 * x124_gate_reverse

        x014_gate = self.sigmoid(self.x014_gatemap(x014))
        x014_gate_reverse = torch.ones(size=x014_gate.size()).cuda() - x014_gate
        x014 = x014 + x014 * x014_gate + x124 * x014_gate_reverse

        x0124 = self.x0124conv(torch.cat((x014, x124), dim=1))
        x1234 = self.x1234conv(torch.cat((x124, x234), dim=1))


        x0124_gate = self.sigmoid(self.x0124_gatemap(x0124))
        x0124_gate_reverse = torch.ones(size=x0124_gate.size()).cuda() - x0124_gate
        x0124 = x0124 + x0124 * x0124_gate + x1234 * x0124_gate_reverse

        x4_guidence = self.x4_guidence_down(x4)
        x4_guidence_out = self.x4_guidence_out(x4_guidence)
        x4_guidence = self.x4_guidence_up(x4_guidence_out)


        feat = torch.cat((x0124, x1234, x4_guidence, x4GAP), dim=1)
        out_feat = self.feature_fusion(feat)

        return out_feat, x4_guidence_out


class DIGModule(nn.Module):
    def __init__(self, stagecur_stage0 = 0, featstagecur = 0, stage_channel = 128):
        super(DIGModule, self).__init__()
        self.stagecur_stage0 = stagecur_stage0
        self.featstagecur = featstagecur

        self.stage0_conv = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

        self.stage_conv = ConvBNReLU(stage_channel, 128, kernel_size=1, stride=1, padding=0)
        self.stage_spatial_conv = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.fusion_conv = ConvBNReLU(384, 256, kernel_size=1, stride=1, padding=0)
        self.globalcontext_conv = ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)

        self.last_conv1 = ConvBNReLU(256, 256, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = ConvBNReLU(384, 256, kernel_size=3, stride=1, padding=1)


        self.GAP = nn.AdaptiveAvgPool2d((15, 15))
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
        stage_global_feat = self.globalcontext_conv(stage_global_feat)


        stage_global_feat = F.interpolate(stage_global_feat, size=(H, W), mode='bilinear', align_corners=True)
        similarity_map = F.cosine_similarity(feat_stagecur_cat, stage_global_feat, dim=1)
        similarity_map = similarity_map.unsqueeze(1)

        feat_stagecur_cat = feat_stagecur_cat * similarity_map


        feat_stagecur_cat = self.last_conv1(feat_stagecur_cat)
        feat_stagecur_cat = torch.cat((feat_stagecur_cat, stagecur), dim=1)
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

        self.backbone = resnet(50, 16)
        self.PAModule = PAmodule(classes)
        self.DIGModule1 = DIGModule(3, 0, 1024)
        self.DIGModule2 = DIGModule(2, 1, 512)
        self.DIGModule3 = DIGModule(1, 1, 256)
        # self.DIGModule4 = DIGModule(0, 1, 128)

        self.feat1_conv = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.feat2_conv = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.feat3_conv = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        self.conv_cat = nn.Sequential(
            ConvBNReLU(448, 256),
            ConvBNReLU(256, 256)
        )

        self.conv_out = nn.Conv2d(256, classes, kernel_size=1, bias=False)

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]
        x1, x2, x3, x4, x0 = self.backbone(x)
        feat, x4_guidence = self.PAModule(x1, x2, x3, x4, x0)
        feat1 = self.DIGModule1(feat, x3, x0)
        feat2 = self.DIGModule2(feat1, x2, x0)
        feat3 = self.DIGModule3(feat2, x1, x0)
        # feat = self.DIGModule4(feat, x0, x0)
        feat1 = self.feat1_conv(feat1)
        feat2 = self.feat2_conv(feat2)
        feat3 = self.feat3_conv(feat3)
        H1, W1 = x1.size()[2:]
        feat = F.interpolate(feat, (H1, W1), mode='bilinear', align_corners=True)
        feat1 = F.interpolate(feat1, (H1, W1), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(feat2, (H1, W1), mode='bilinear', align_corners=True)
        feat = torch.cat((feat, feat1, feat2, feat3), dim=1)
        feat = self.conv_cat(feat)
        final = self.conv_out(feat)

        x4_guidence = F.interpolate(x4_guidence, (H, W), mode='bilinear', align_corners=True)
        final = F.interpolate(final, (H, W), mode='bilinear', align_corners=True)

        return final, x4_guidence



