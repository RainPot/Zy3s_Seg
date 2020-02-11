import numpy as np
import torch
import config_CS as config
# import config_ADE20K as config

def fast_hist(label_true, label_pred):
    n_classes = config.classes
    mask = (label_true >= 0) & (label_true < n_classes)
    hist = torch.bincount(
        n_classes * label_true[mask].int() + label_pred[mask].int(),
        minlength = n_classes ** 2,
    ).reshape(n_classes, n_classes)
    return hist

label_names = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def cal_scores(hist):
    n_classes = config.classes
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq >0]).sum()
    cls_iu = dict(zip(label_names, iu))

    return mean_iu


