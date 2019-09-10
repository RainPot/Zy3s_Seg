import torch
from torch.utils.data import Dataset
# from torchvision.datasets.Cityscapes
from torchvision.transforms import transforms
import os
import numpy as np
import config
from datasets.transform import *


class cityscapestrain(Dataset):
    def __init__(self, mode='train', ignore_label=255):
        super(cityscapestrain, self).__init__()
        self.mode = mode
        self.image = []
        self.label = []

        image_dir = os.path.join('./leftImg8bit', self.mode)
        mask_dir = os.path.join('./gtFine', self.mode)

        for img_folders in os.listdir(image_dir):
            im_names = os.listdir(os.path.join(image_dir, img_folders))
            imgpths = [os.path.join(image_dir, img_folders, el) for el in im_names]
            self.image.extend(imgpths)

        for mask_folders in os.listdir(mask_dir):
            mask_names = os.listdir(os.path.join(mask_dir, mask_folders))
            maskpths = []
            for el in mask_names:
                # if el[-12:-3] == 'labelIds.':
                #     maskpths.append(os.path.join(mask_dir, mask_folders, el))
                maskpths.append(os.path.join(mask_dir, mask_folders, el))
            self.label.extend(maskpths)

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        print('{} images are loaded!'.format(len(self.image)))

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std)
        ])

        self.trans = Compose([
            ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation
            ),
            HorizontalFlip(),
            RandomScale(config.scales),
            Crop(config.crop_size)
        ])

    def __len__(self):
        return len(self.image)

    def ID2TrainID(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        imagename = self.image[index]
        labelname = self.label[index]
        image = Image.open(imagename)
        label = Image.open(labelname)

        if self.mode == 'train':
            data = dict(im=image, lb=label)
            data = self.trans(data)
            image, label = data['im'], data['lb']

        imgs = self.to_tensor(image)
        #label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = np.array(label).astype(np.int64)
        # label = self.ID2TrainID(label)
        return imgs, label




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    ds = cityscapestrain(mode='train')
    dl = DataLoader(ds,
                    batch_size=4,
                    num_workers=4,
                    shuffle=True,
                    drop_last=True)
    for imgs, lbs in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break





