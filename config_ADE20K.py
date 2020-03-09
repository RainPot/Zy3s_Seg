# train
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
brightness = 0.5
contrast = 0.5
saturation = 0.5
scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# crop_size = (864, 864)
crop_size = (512, 512)
# imgs_per_gpu = 2
imgs_per_gpu = 4
# max_iter = 70000
max_iter = 150000
# classes = 19
classes = 150
# ignore_label = 255
ignore_label = -1


warmup_steps = 1000
warmup_start_lr = 5e-6
# lr_start = 1e-2
lr_start = 2e-2
momentum = 0.9
# weight_decay = 5e-4
weight_decay = 1e-4
lr_power = 0.9
ohem_thresh = 0.7



eval_scales = (0.5, 0.75, 1.0, 1.25)
eval_flip = True


LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4


port = 35770


