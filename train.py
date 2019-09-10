import torch
import torch.nn as nn
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import SGD
from datasets.cityscapes import cityscapestrain
from model.origin_res import Origin_Res
from model.deeplabv3 import Deeplab_v3plus
import argparse
import config
from pallete import get_mask_pallete



class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.NLLLoss = nn.NLLLoss(weight=None, reduction='none', ignore_index=255)

    def forward(self, output, label):
        a = F.log_softmax(output, dim=1)
        loss = self.NLLLoss(F.log_softmax(output, dim=1), label)
        return loss.mean(dim=2).mean(dim=1)

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
                if m[1].bias is not None:
                    yield m[1].bias

def poly_lr_scheduler(opt, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    opt.param_groups[0]['lr'] = 1 * new_lr
    opt.param_groups[1]['lr'] = 1 * new_lr
    opt.param_groups[2]['lr'] = 2 * new_lr



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
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=2,
                            pin_memory=True,
                            drop_last=True)

    # net = Origin_Res()
    net = Deeplab_v3plus()
    net.train()
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank, ],
                                              output_device=args.local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    criterion = Criterion().cuda()

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


    total_loss = 0
    n_epoch = 0

    data = iter(dataloader)
    for i in range(config.max_iter):
        try:
            image, label = next(data)
        except:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
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
        image_see = image.cpu().numpy()
        label = label.cuda()
        label_see = label.cpu().numpy()

        output = net(image)
        output_see = output.detach().cpu().numpy()

        predict = torch.max(output[0], 1)[1].cpu().numpy() + 1

        mask = get_mask_pallete(predict, 'cityscapes')


        loss = criterion(output, label)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i+1) % 10 == 0 and dist.get_rank() == 0:
            print('iter: {}, loss: {}'.format(i+1, total_loss / 100.0))
            total_loss = 0

        if (i+1) % 500 == 0 and dist.get_rank() == 0:
            torch.save(net.state_dict(), './Res{}.pth'.format(i+1))


if __name__ == '__main__':
    args = parse_args()
    train(args)
