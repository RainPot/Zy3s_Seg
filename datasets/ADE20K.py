#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import config_ADE20K as cfg
from datasets.transform import *



class ADE20K(Dataset):
    def __init__(self, mode='train', *args, **kwargs):
        super(ADE20K, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val')
        self.mode = mode

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory


        if self.mode == 'train':
            self.names = [el for el in os.listdir(osp.join('./ADEChallengeData2016', 'images', 'training'))]

        if self.mode == 'val':
            self.names = [el for el in os.listdir(osp.join('./ADEChallengeData2016', 'images', 'validation'))]

        ## parse gt directory

        self.len = len(self.names)
        # assert set(self.imgnames) == set(self.lbnames)


        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
            ])
        self.trans = Compose([
            ColorJitter(
                brightness = cfg.brightness,
                contrast = cfg.contrast,
                saturation = cfg.saturation),
            HorizontalFlip(),
            RandomScale(cfg.scales),
            RandomCrop(cfg.crop_size)
            ])


    def __getitem__(self, idx):
        name = self.names[idx]
        if self.mode == 'train':
            impth = osp.join('./ADEChallengeData2016', 'images', 'training', name)
            lbpth = osp.join('./ADEChallengeData2016', 'annotations', 'training', name[:-4]+'.png')
        elif self.mode == 'val':
            impth = osp.join('./ADEChallengeData2016', 'images', 'validation', name)
            lbpth = osp.join('./ADEChallengeData2016', 'annotations', 'validation', name[:-4]+'.png')
        else:
            print('nonono')
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']

        imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :] - 1
        #label = self.convert_labels(label)
        return imgs, label, name


    def __len__(self):
        return self.len



class ADE20K_test(Dataset):
    def __init__(self, mode='test', *args, **kwargs):
        super(ADE20K_test, self).__init__(*args, **kwargs)
        self.mode = mode

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory

        self.names = [el for el in os.listdir(osp.join('./release_test', 'testing'))]


        ## parse gt directory

        self.len = len(self.names)
        # assert set(self.imgnames) == set(self.lbnames)


        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
            ])
        self.trans = Compose([
            ColorJitter(
                brightness = cfg.brightness,
                contrast = cfg.contrast,
                saturation = cfg.saturation),
            HorizontalFlip(),
            RandomScale(cfg.scales),
            RandomCrop(cfg.crop_size)
            ])


    def __getitem__(self, idx):
        name = self.names[idx]

        impth = osp.join('./release_test', 'testing', name)


        img = Image.open(impth).convert('RGB')

        imgs = self.to_tensor(img)

        # label = self.convert_labels(label)
        return imgs, name



    def __len__(self):
        return self.len