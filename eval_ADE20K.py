import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets.cityscapes import CityScapes
from datasets.ADE20K import ADE20K
from model.v8cADE import HighOrder
from ablationstudy.ADE20KGPNettest import PANet
from metric import fast_hist, cal_scores
import config_ADE20K as config
import argparse
import numpy as np
from PIL import Image


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest='local_rank',
        type=int,
        default=-1
    )
    return parse.parse_args()


ignore_label = -1

label_mapping = {-1: ignore_label, 0: ignore_label,
                 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label,
                 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label,
                 10: ignore_label, 11: 2, 12: 3,
                 13: 4, 14: ignore_label, 15: ignore_label,
                 16: ignore_label, 17: 5, 18: ignore_label,
                 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15,
                 29: ignore_label, 30: ignore_label,
                 31: 16, 32: 17, 33: 18}


def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def eval(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(config.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
        # rank=args.local_rank
    )

    dataset = ADE20K(mode='val')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    net = PANet(150)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank],
                                              output_device=args.local_rank)
    net.load_state_dict(torch.load('./GPADE20Kres50150000.pth', map_location='cpu'))
    net.eval()

    data = iter(dataloader)
    palette = get_palette(256)
    num = 0
    hist = 0
    with torch.no_grad():
        while 1:
            try:
                image, label, name = next(data)
            except:
                break

            image = image.cuda()
            label = label.cuda()
            label = torch.squeeze(label, 1)
            N, _, H, W = image.size()
            preds = torch.zeros((N, 150, H, W))
            preds = preds.cuda()
            for scale in config.eval_scales:
                new_hw = [int(H * scale), int(W * scale)]
                image_change = F.interpolate(image, new_hw, mode='bilinear', align_corners=True)
                output, w = net(image_change)
                output = F.interpolate(output, (H, W), mode='bilinear', align_corners=True)
                output = F.softmax(output, 1)
                preds += output
                if config.eval_flip:
                    output, w = net(torch.flip(image_change, dims=(3,)))
                    output = torch.flip(output, dims=(3,))
                    output = F.interpolate(output, (H, W), mode='bilinear', align_corners=True)
                    output = F.softmax(output, 1)
                    preds += output
            pred = preds.max(dim=1)[1]
            hist_once = fast_hist(label, pred)
            hist = torch.tensor(hist).cuda()
            hist = hist + hist_once
            dist.all_reduce(hist, dist.ReduceOp.SUM)
            num += 1
            if num % 5 == 0:
                print('iter: {}'.format(num))

            preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
            # for i in range(preds.shape[0]):
            #     pred = convert_label(preds[i], inverse=True)
            #     save_img = Image.fromarray(pred)
            #     save_img.putpalette(palette)
            #     save_img.save(os.path.join('./CS_results/', name[i] + '.png'))

        hist = hist.cpu().numpy().astype(np.float32)
        miou = cal_scores(hist)

    print('miou = {}'.format(miou))


if __name__ == '__main__':
    args = parse_args()
    eval(args)









