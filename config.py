# train
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
brightness = 0.5
contrast = 0.5
saturation = 0.5
scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
crop_size = (640, 640)
max_iter = 60000
classes = 19

eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
eval_flip = True


LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4


port = 35768


