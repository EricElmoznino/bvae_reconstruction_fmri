import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageFilesDataset(Dataset):

    def __init__(self, image_paths, training=False, nc=3):
        super().__init__()
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
        image = Image.open(self.data[item])
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)


class ImageFolderDataset(ImageFilesDataset):

    def __init__(self, image_dir, training=False, nc=3):
        image_paths = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, filename) for filename in image_paths]
        super().__init__(image_paths, training, nc)


class NumpyImageDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        data = np.load(data_path, encoding='bytes')['imgs']
        self.data = torch.from_numpy(data).unsqueeze(1).float() / 255
        self.nc = 1

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.size(0)
