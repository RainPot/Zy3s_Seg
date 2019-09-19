import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets.cityscapes import CityScapes
from model.origin_res import Origin_Res
from model.deeplabv3 import Deeplab_v3plus
from metric import fast_hist, cal_scores
import config
import argparse
import numpy as np

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest = 'local_rank',
        type = int,
        default = -1,
    )
    return parse.parse_args()


def eval(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(config.port),
        world_size=torch.cuda.device_count(),
        rank=0
    )

    dataset = CityScapes(mode='val')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    net = Origin_Res()
    # net = Deeplab_v3plus()
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank],
                                              output_device=args.local_rank)
    net.load_state_dict(torch.load('./Res1500.pth'))
    net.eval()

    data = iter(dataloader)
    num = 0
    hist = 0
    with torch.no_grad():
        while 1:
            try:
                image,label = next(data)
            except:
                break
            image = image.cuda()
            label = label.cuda()
            label = torch.squeeze(label, 1)

            output = net(image)
            pred = output.max(dim=1)[1]
            hist_once = fast_hist(label, pred)
            hist = torch.tensor(hist).cuda()
            hist = hist + hist_once
            dist.all_reduce(hist, dist.ReduceOp.SUM)
            num += 1
            if num % 50 == 0:
                print('iter :{}'.format(num))
        hist = hist.cpu().numpy().astype(np.float32)
        miou = cal_scores(hist)

    print('miou = {}'.format(miou))





if __name__ == '__main__':
    args = parse_args()
    eval(args)
