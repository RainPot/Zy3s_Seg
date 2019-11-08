import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image
import cv2
import numpy as np


import torch.distributed as dist

from model.highorderv8 import HighOrder



if __name__ == '__main__':

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend = 'nccl',
        init_method='tcp://127.0.0.1:35789',
        world_size=torch.cuda.device_count(),
        rank=0
    )


    net = HighOrder(19)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[0],
                                              output_device=0)

    net.load_state_dict(torch.load('./Res60000.pth'))
    net.eval()

    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open('./leftImg8bit/test/berlin/berlin_000040_000019_leftImg8bit.png')
    image = to_tensor(image)
    image = torch.unsqueeze(image, 0)

    print('start predict!')
    output, camx1, camx4 = net(image)



    camx4 = torch.squeeze(camx4, 0)
    # camx4 = camx4.sum(dim=0)
    channel_1 = camx4[100].cpu().detach().numpy()
    print('down')

    height, width = image.size()[2], image.size()[3]
    cam_img = (channel_1 - channel_1.min()) / (channel_1.max() - channel_1.min())
    cam_img = np.array(255 * cam_img).astype(np.uint8)
    print('start generate CAM')
    heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
    cv2.imwrite('./cam.jpg', heatmap)
