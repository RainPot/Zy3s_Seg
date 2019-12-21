import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
import numpy as np
from PIL import Image
import config_VOC as cfg
from datasets.transform import *

class VOCContext(Dataset):
    def __init__(self, mode='train', *args, **kwargs):
        super(VOCContext, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode

        self.imgsname = []
        self.imgspth = []
        imgfilepth = './VOC2010/JPEGImages/'
        self.lbspth = []
        lbfilepth = './VOC2010/SegmentationClass/'
        F = open(osp.join('./VOC2010/ImageSets/Segmentation/', 'train.txt'), 'r')
        for line in F.readlines():
            self.imgspth.append(osp.join(imgfilepth, line.strip()) + '.jpg')
            self.lbspth.append(osp.join(lbfilepth, line.strip()) + '.png')
            self.imgsname.append(line.strip())


        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std)
        ])

        self.trans = Compose([
            ColorJitter(
                brightness = cfg.brightness,
                contrast = cfg.contrast,
                saturation = cfg.saturation
            ),
            HorizontalFlip(),
            RandomScale(cfg.scales),
            RandomCrop(cfg.crop_size)
        ])

    def __getitem__(self, idx):
        fn = self.imgsname[idx]
        impth = self.imgspth[idx]
        lbpth = self.lbspth[idx]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]

        return imgs, label, fn

    def __len__(self):
        return len(self.imgspth)








