# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

import os
import glob


def _load_images(data_root):
    # root
    rgb_root = os.path.join(data_root, 'image')
    mask_root = os.path.join(data_root, 'mask')

    image_ls = []
    mask_ls = []
    for image_path in glob.glob(os.path.join(rgb_root, '*.png')):
        image_name = image_path.split('/')[-1]
        #print(image_name)

        mask_path = os.path.join(mask_root, image_name)

        image_ls.append(image_path)
        mask_ls.append(mask_path)

    return image_ls, mask_ls

class PathologyDataset(Dataset):
    def __init__(self, data_root, transform, target=3):
        self.data, self.label = _load_images(data_root)
        self.transform = transform
        self.target = target
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        mask = Image.open(self.label[idx])

        image = self.transform(image)
        mask = self.transform(mask)
        one_hot_vector = [torch.ones(1, 512, 512) for _ in range(self.target)]
        one_hot_vector = torch.cat(one_hot_vector, axis=0)

        return image, mask, one_hot_vector

def get_dataloader(data_root, batch_size, transform, resize_shape=None):
    dataset = PathologyDataset(data_root, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader 

if __name__ == '__main__':
    DATASET_ROOT = './pathology_dataset'
    TRRANSFORM = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL images to tensors
    ])

    dataloader = get_dataloader(DATASET_ROOT, 4, TRRANSFORM)

    for x, y, z in dataloader:
        print(x.shape, y.shape, z.shape)