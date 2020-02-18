import torch
import torch.nn as nn
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import SGD
from datasets.cityscapes import CityScapes, CityScapes_trainval
from datasets.ADE20K import ADE20K
from model.origin_res import Origin_Res
from model.deeplabv3 import Deeplab_v3plus
from model.v8c import HighOrder
from model.PANet13 import PANet
#from model.baseline import HighOrder
import argparse
import config_CS as config
# import config_ADE20K as config
from pallete import get_mask_pallete
import time
from loss import OhemCELoss
from optimizer import Optimizer



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



def train(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:34640',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
        # rank=0
    )

    # dataset = CityScapes(mode='train')
    dataset = CityScapes_trainval(mode='train')
    # dataset = ADE20K(mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=config.imgs_per_gpu,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    print(dataloader.__len__())
    # net = Origin_Res()
    net = PANet(config.classes)
    # net = HighOrder(config.classes)
    # for i in net.named_modules():
    #     print(i)
    # net = Deeplab_v3plus()

    net.train()
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank],
                                              output_device=args.local_rank)


    n_min = config.imgs_per_gpu * config.crop_size[0] * config.crop_size[1] // 16
    criteria = OhemCELoss(thresh=config.ohem_thresh, n_min=n_min, ignore_lb=config.ignore_label).cuda()

    optimizer = Optimizer(
        net,
        config.lr_start,
        config.momentum,
        config.weight_decay,
        config.warmup_steps,
        config.warmup_start_lr,
        config.max_iter,
        config.lr_power
    )


    total_loss = 0
    n_epoch = 0

    data = iter(dataloader)
    for i in range(config.max_iter):
        start = time.time()
        try:
            image, label, name = next(data)
        except:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
            data = iter(dataloader)
            image, label, name = next(data)


        image = image.cuda()
        image_see = image.cpu().numpy()
        label = label.cuda()
        label = torch.squeeze(label, 1)

        label_see = label.cpu().numpy()

        output = net(image)
        output_see = output.detach().cpu().numpy()

        predict = torch.max(output[0], 1)[1].cpu().numpy() + 1

        mask = get_mask_pallete(predict, 'cityscapes')


        loss = criteria(output, label)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i+1) % 100 == 0 and dist.get_rank() == 0:
            end = time.time()
            once_time = end - start
            remaining_step = config.max_iter - i
            remaining_time = once_time * remaining_step
            m, s = divmod(remaining_time, 60)
            h, m = divmod(m, 60)
            print('iter: {}, loss: {}, time: {}h:{}m'.format(i+1, total_loss / 100.0, int(h), int(m)))
            total_loss = 0

        if (i+1) % 100 == 0 and (i+1) >= (int(config.max_iter) - 200) and dist.get_rank() == 0:
            torch.save(net.state_dict(), './PANet14_trainval{}.pth'.format(i+1))



if __name__ == '__main__':
    args = parse_args()
    train(args)
