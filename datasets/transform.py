from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random


class Crop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        im = data['im']
        lb = data['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size
        if (W, H) == (w, h):
            return dict(im = im, lb = lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(w * scale + 1), int(h * scale + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw + W), int(sh + H)
        return dict(
            im = im.crop(crop),
            lb = lb.crop(crop)
        )


class HorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        else:
            im = data['im']
            lb = data['lb']
            return dict(
                im = im.transpose(Image.FLIP_LEFT_RIGHT),
                lb = lb.transpose(Image.FLIP_LEFT_RIGHT)
            )

class RandomScale(object):
    def __init__(self, scale=(1, )):
        self.scales = scale

    def __call__(self, data):
        im = data['im']
        lb = data['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(
            im = im.resize((w, h), Image.BILINEAR),
            lb = lb.resize((w, h), Image.NEAREST)
        )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W * ratio), int(H * ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, data):
        im = data['im']
        lb = data['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(
            im = im,
            lb = lb
        )

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, data):
        for comp in self.do_list:
            data = comp(data)
        return data


