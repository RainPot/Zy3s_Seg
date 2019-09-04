import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets.cityscapes import cityscapestrain
from backbone.resnet import resnet
import argparse
import config


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest = 'local_rank',
        type = int,
        default = -1,
    )
    return parse.parse_args()

def criterion():
    pass

def train(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(config.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )

    dataset = cityscapestrain(mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(cityscapestrain,
                            batch_size=1,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    net = resnet(101, 16)
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank,],
                                              output_device=args.local_rank)







if __name__ == '__main__':
    args = parse_args()
    train(args)




