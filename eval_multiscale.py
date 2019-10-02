import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets.cityscapes import CityScapes
from model.highorder import HighOrder
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
        default = -1
    )
    return parse.parse_args()

def eval(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend = 'nccl',
        init_method = 'tcp://127.0.0.1:{}'.format(config.port),
        world_size = torch.cuda.device_count(),
        rank=0
    )

    dataset = CityScapes(mode='val')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        sampler = sampler,
        num_workers = 4,
        drop_last = False,
        pin_memory = True
    )

    net = HighOrder(19)
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank],
                                              output_device=args.local_rank)
    net.load_state_dict(torch.load(''))
    net.eval()


    

