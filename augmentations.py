import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision.transforms.functional as tf

from torchvision.transforms import RandomGrayscale, RandomAffine

## My data augmentation
class GrayScale(object):
    def __init__(self, p):
        self.gray_scale = RandomGrayscale(p)
        
    def __call__(self, img, mask):
        assert img.size == mask.size
        return self.gray_scale(img), mask

class RandomZoomOut(object):
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, img, mask):
        assert img.size == mask.size

        # 生成隨機縮放比例
        random_scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # 將隨機縮放應用於圖像和掩碼
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=random_scale,
                angle=0,
                fill=(0.0, 0.0, 0.0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=random_scale,
                angle=0,
                fill=(0),
                shear=0.0,
            ),
        )
        
class RandomCombineImage(object):
    def __init__(self, p=0.4, translate1=(0, 0), translate2=(0.5, 0.3), scale1=(0.6, 0.8),  scale2=(0.4, 0.55)):
        self.p = p
        # self.translate1 = translate1
        # self.translate2 = translate2
        # self.scale1 = scale1
        # self.scale2 = scale2

        self.affine_transfomer_main = RandomAffine(degrees=(-5, 5), translate=translate1, scale=scale1)
        self.affine_transfomer_sec = RandomAffine(degrees=(-5, 5), translate=translate2, scale=scale2)
        
    def __call__(self, img, mask):
        if random.random() < self.p:
            size = mask.size()  
            
            B, _, H, W = img.size()
            numbers = list(range(B))
            random.shuffle(numbers)
            
            sec_img = img[numbers]
            sec_mask = mask[numbers]
            
            affine_params = self.affine_transfomer_main.get_params(
                self.affine_transfomer_main.degrees,
                self.affine_transfomer_main.translate,
                self.affine_transfomer_main.scale,
                self.affine_transfomer_main.shear,
                (H, W)
            )

            img = tf.affine(img, *affine_params, fill=(-1.0, -1.0, -1.0))
            mask = tf.affine(mask, *affine_params, fill=0)
            
            main_mask = mask[:, :, :].view(size[0], 1, size[1], size[2])
            main_mask = (main_mask == 0).any(dim=1, keepdim=True).float()
            main_mask = main_mask.repeat(1, 3, 1, 1)
            
            affine_sec_params = self.affine_transfomer_sec.get_params(
                self.affine_transfomer_sec.degrees,
                self.affine_transfomer_sec.translate,
                self.affine_transfomer_sec.scale,
                self.affine_transfomer_sec.shear,
                (H, W)
            )

            sec_img = tf.affine(sec_img, *affine_sec_params, fill=(-1.0, -1.0, -1.0))
            sec_mask = tf.affine(sec_mask, *affine_sec_params, fill=0)
            sec_mask = sec_mask[:, :, :].view(size[0], 1, size[1], size[2])
            sec_mask = (sec_mask == 0).any(dim=1, keepdim=True).float()
            sec_mask = sec_mask.repeat(1, 3, 1, 1)
            sec_img = main_mask * (1-sec_mask) * sec_img
            i_mask = 1 - main_mask + sec_mask * main_mask
            
            img = img * i_mask + sec_img
            
        return img, mask

class CenterCrop(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        B, C, H, W = img.size()
        h, w = self.size
        del_h = (H - h) // 2
        del_w = (W - w) // 2

        img[:, :, 0:del_h, :] = -1.0
        img[:, :, del_h+H:H, :] = -1.0
        img[:, :, :, 0:del_w] = -1.0
        img[:, :, :, del_w+W:W] = -1.0

        return img
# ----------------------------------------

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True
        # assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask

class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(
                1 - self.saturation, 1 + self.saturation)),
            mask,
        )

class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask

class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            # return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))
            # Need to pay attention to the index problem !!!
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask_copy = np.array(mask).copy()

            right_idx = [5, 7, 9]
            left_idx = [4, 6, 8]

            for i in range(3):
                right_pos = np.where(mask_copy == right_idx[i])
                left_pos = np.where(mask_copy == left_idx[i])
                mask_copy[right_pos[0], right_pos[1]] = left_idx[i]
                mask_copy[left_pos[0], left_pos[1]] = right_idx[i]
            return img, Image.fromarray(mask_copy)

        return img, mask

class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask

class FreeScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        return (img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size), Image.NEAREST))

class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
            tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fill=(0),
            ),
        )

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        assert img.size == mask.size
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                fill=(0.0, 0.0, 0.0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                fill=(0),
                shear=0.0,
            ),
        )

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2.0)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (img.resize((w, h), Image.BILINEAR),
                     mask.resize((w, h), Image.NEAREST))

        return self.crop(*self.scale(img, mask))


def img_transform(img):
    # 0-255 to 0-1
    # img = np.float32(np.array(img)) / 255.
    # img = img.transpose((2, 0, 1))
    # img = torch.from_numpy(img.copy())
    import torchvision.transforms as transforms
    transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transformer(img)

    return img


def mask_transform(segm):
    # to tensor
    segm = torch.from_numpy(np.array(segm)).long()

    return segm


def edge_contour(label, edge_width=3):
    import cv2 as cv

    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (edge_width, edge_width))
    edge = cv.dilate(edge, kernel)

    return edge
