import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import utils


class ImageFilesDataset(Dataset):

    def __init__(self, image_paths, training=False):
        super().__init__()
        assert len(image_paths) > 0
        random.shuffle(image_paths)
        self.image_paths = image_paths
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
        self.nc = self[0].size(0)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


class ImageFolderDataset(ImageFilesDataset):

    def __init__(self, image_dir, training=False):
        image_paths = utils.recursive_folder_image_paths(image_dir)
        super().__init__(image_paths, training)


class NumpyImageDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        data = np.load(data_path, encoding='bytes')['imgs']
        assert len(data) > 0
        np.random.shuffle(data)
        data = torch.from_numpy(data).float()
        if data.ndim == 2:
            data = data.unsqueeze(0)
        self.data = data
        self.nc = data.size(0)

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
