import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import resnet



class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ):
        super(ConvBNReLU, self).__init__()
