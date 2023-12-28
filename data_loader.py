import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp
import glob
from augmentations import *


class CelebAMaskHQ(Dataset):
    """The data reading method modified from the original paper"""

    def __init__(self, img_path, label_path, resize=512, transform=None, mode=True):
        self.img_path = img_path
        self.label_path = label_path
        self.train_dataset = []
        self.test_dataset = []
        self.transform = transform
        self.mode = mode
        self.resize = resize
        self.preprocess()

        if mode:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.img_path) if osp.isfile(osp.join(self.img_path, name))])):
            img_path = osp.join(self.img_path, str(i)+'.jpg')
            label_path = osp.join(self.label_path, str(i)+'.png')

            if self.mode:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]

        # Uniform image size
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # Image resized to the same dimension
        image, label = Compose(
            [FreeScale(self.resize)])(image, label)

        if self.mode: # train mode
            if self.transform is not None:
                image, label = self.transform(image, label)

            # Convert it to pytorch style
            image = img_transform(image)
            mask = mask_transform(label)
            edge = mask_transform(Image.fromarray(edge_contour(np.asarray(label))))

            return image, mask, edge

        else:
            image = img_transform(image)
            mask = mask_transform(label)

            return image, mask

    def __len__(self):
        return self.num_images


class CustomDataLoader:
    def __init__(self, img_path, label_path, image_size, batch_size, num_workers, transform=None, mode=True):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.transform = transform

    def loader(self):
        dataset = CelebAMaskHQ(
            self.img_path, self.label_path, self.imsize, self.transform, self.mode)

        if self.mode == True:
            loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self.batch,
                                                 shuffle=True,
                                                 num_workers=self.num_workers,
                                                 drop_last=True,
                                                 pin_memory=True)
            return loader

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=False,
                                             num_workers=self.num_workers,
                                             drop_last=True,
                                             pin_memory=True)
        return loader


class CelebAMaskHQ_Unseen(Dataset):
    def __init__(self, img_path, resize=512, transform=None):
        self.img_path = img_path
        self.dataset = []
        self.transform = transform
        self.resize = resize
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        def custom_sort(item):
            match = re.findall(r'(\d+)', item)
            if match and len(match) >= 2:
                return int(match[0]), int(match[1])
            return 0, 0
        self.dataset.extend(glob.glob(osp.join(self.img_path, "*.jpg")))
        self.dataset = sorted(self.dataset, key=custom_sort)


    def __getitem__(self, index):
        img_path = self.dataset[index]
        # Uniform image size
        image = Image.open(img_path).convert("RGB")
        # Image resized to the same dimension
        image = image.resize((self.resize, self.resize), Image.BILINEAR)
        image = img_transform(image)
        return image

    def __len__(self):
        return self.num_images
    
class Unseen_DataLoader:
    def __init__(self, img_path, image_size, batch_size, num_workers, transform=None):
        self.img_path = img_path
        self.imsize = image_size
        self.batch = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def loader(self):
        dataset = CelebAMaskHQ_Unseen(
            self.img_path, self.imsize, self.transform)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=False,
                                             num_workers=self.num_workers,
                                             drop_last=True,
                                             pin_memory=True)
        return loader

