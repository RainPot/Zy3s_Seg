import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
import cv2
import numpy as np


import torch.distributed as dist

# from model.highorderv8 import HighOrder
from ablationstudy.visualization import PANet
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest = 'local_rank',
        type = int,
        default = -1
    )
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dist.init_process_group(
        backend = 'nccl',
        init_method='tcp://127.0.0.1:35789',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )


    net = PANet(19)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank],
                                              output_device=args.local_rank)

    net.load_state_dict(torch.load('./PANet20_train60000.pth'))
    net.eval()

    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open('./leftImg8bit/val/lindau/lindau_000057_000019_leftImg8bit.png')
    image = to_tensor(image)
    image = torch.unsqueeze(image, 0)

    print('start predict!')
    output = net(image)

    i=0
    for feature in output[2:]:

        camx4 = torch.squeeze(feature, 0)
        # camx4 = camx4.softmax(dim=0)

        print(camx4)

        channel_1 = camx4[0].cpu().detach().numpy()
        print('down')

        height, width = image.size()[2], image.size()[3]
        # cam_img = (channel_1 - channel_1.min()) / (channel_1.max() - channel_1.min())
        # channel_1 = (channel_1 - channel_1.min()) / (channel_1.max() - channel_1.min())
        cam_img = np.array(255 * channel_1).astype(np.uint8)
        print('start generate CAM')
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
        # heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        cv2.imwrite('./CAMcam'+str(i)+'.jpg', heatmap)
        i += 1
