import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import SGD
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

def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight

    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], nn.SyncBatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight

    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.SyncBatchNorm):
                if m[1].weight is not None:
                    yield m[1].bias

def poly_lr_scheduler(opt, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    opt.param_groups[0]['lr'] = 1 * new_lr
    opt.param_groups[1]['lr'] = 1 * new_lr
    opt.param_groups[2]['lr'] = 2 * new_lr


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

    optimizer = SGD(
        params=[
            {
                'params': get_params(net, key='1x'),
                'lr': 1 * config.LR,
                'weight_decay': config.WEIGHT_DECAY,
            },
            {
                'params': get_params(net, key='1y'),
                'lr': 1 * config.LR,
                'weight_decay': 0,
            },
            {
                'params': get_params(net, key='2x'),
                'lr': 2 * config.LR,
                'weight_decay': 0.0
            }
        ],
        momentum=config.LR_MOM
    )

    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank,],
                                              output_device=args.local_rank)
    net.train()
    data = iter(dataloader)
    for i in range(config.max_iter):
        try:
            image, label = next(data)
        except:
            data = iter(dataloader)
            image, label = next(data)

        poly_lr_scheduler(
            opt=optimizer,
            init_lr=config.LR,
            iter=i+1,
            lr_decay_iter=config.LR_DECAY,
            max_iter=config.max_iter,
            power=config.POLY_POWER
        )

        image = image.cuda()
        label = label.cuda()

        output = net(image)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()







if __name__ == '__main__':
    args = parse_args()
    train(args)




