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
import config_CS as cfg
from datasets.transform import *



class CityScapes(Dataset):
    def __init__(self, mode='train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode


        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join('./leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join('./gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

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
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        #label = self.convert_labels(label)
        return imgs, label, fn


    def __len__(self):
        return self.len



class CityScapes_trainval(Dataset):
    def __init__(self, mode='train', *args, **kwargs):
        super(CityScapes_trainval, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join('./leftImg8bit', 'train')
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        impth = osp.join('./leftImg8bit', 'val')
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join('./gtFine', 'train')
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        gtpth = osp.join('./gtFine', 'val')
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelTrainIds' in el]
            names = [el.replace('_gtFine_labelTrainIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

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
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        #label = self.convert_labels(label)
        return imgs, label, fn


    def __len__(self):
        return self.len



class CityScapes_test(Dataset):
    def __init__(self, *args, **kwargs):
        super(CityScapes_test, self).__init__(*args, **kwargs)

        self.mode = 'test'


        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join('./leftImg8bit', self.mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))



        self.imnames = imgnames
        self.len = len(self.imnames)

        assert set(self.imnames) == set(self.imgs.keys())


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
        fn  = self.imnames[idx]
        impth = self.imgs[fn]

        img = Image.open(impth)

        imgs = self.to_tensor(img)

        #label = self.convert_labels(label)
        return imgs, fn


    def __len__(self):
        return self.len




if __name__ == "__main__":
    # from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break