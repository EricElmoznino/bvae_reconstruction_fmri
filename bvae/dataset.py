import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from bvae import utils


class ImageFilesDataset(Dataset):

    def __init__(self, image_paths, training=False, nc=3):
        super().__init__()
        random.shuffle(image_paths)
        self.image_paths = image_paths
        self.nc = nc
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        if self.nc == 3:
            image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


class ImageFolderDataset(ImageFilesDataset):

    def __init__(self, image_dir, training=False, nc=3):
        image_paths = utils.recursive_folder_image_paths(image_dir)
        super().__init__(image_paths, training, nc)


class NumpyImageDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        data = np.load(data_path, encoding='bytes')['imgs']
        np.random.shuffle(data)
        self.data = torch.from_numpy(data).unsqueeze(1).float()
        self.nc = 1

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.size(0)


class DebugDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.nc = 3

    def __getitem__(self, item):
        return torch.rand(3, 64, 64)

    def __len__(self):
        return 36
