from PIL import Image
import numpy as np

def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

ignore_label = -1

label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


palette = get_palette(256)
# save_img = Image.open('D:/dataset/cityscapes_results/valgtFine/lindau/lindau_000034_000019_gtFine_labelTrainIds.png')
gt = np.array(Image.open('D:/dataset/cityscapes_results/valgtFine/frankfurt/frankfurt_000001_013710_gtFine_labelTrainIds.png'))
gt = convert_label(gt, inverse=True)
gt = Image.fromarray(gt)
gt.putpalette(palette)
gt.save('D:/dataset/cityscapes_results/groundtruthownpallete/frankfurt_000001_013710_gtFine_labelTrainIds.png')