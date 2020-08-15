import os
import math
import imghdr
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.transforms import functional as transforms
from torchvision.utils import make_grid
from bvae.model import BetaVAE


def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def load_image(image_file, resolution=64):
    image = Image.open(image_file).convert('RGB')
    assert image.width == image.height
    image = transforms.resize(image, resolution)
    image = transforms.to_tensor(image)
    return image


def recursive_folder_image_paths(folder_path):
    file_paths = []
    for dirpath, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            if imghdr.what(file_path) is not None:
                file_paths.append(file_path)
    return file_paths


def combine_image_folders(folder_paths, save_folder_path, resolution):
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    for folder in folder_paths:
        for dirpath, dirs, files in os.walk(folder):
            for filename in files:
                if os.path.exists(os.path.join(save_folder_path, filename)):
                    continue
                file_path = os.path.join(dirpath, filename)
                try:
                    image = Image.open(file_path)
                except UnidentifiedImageError:
                    continue
                image = resize_and_square_image(image, resolution)
                image.save(os.path.join(save_folder_path, filename))


def resize_and_square_image(image, resolution):
    assert isinstance(resolution, int)
    if image.width < image.height:
        image = transforms.resize(image, (resolution, int(resolution * image.width / image.height)))
        dif = image.height - image.width
        image = transforms.pad(image, (math.floor(dif/2), 0, math.ceil(dif/2), 0), fill=(255, 255, 255))
    elif image.width > image.height:
        image = transforms.resize(image, (int(resolution * image.height / image.width), resolution))
        dif = image.width - image.height
        image = transforms.pad(image, (0, math.floor(dif/2), 0, math.ceil(dif/2)), fill=(255, 255, 255))
    else:
        image = transforms.resize(image, resolution)
    return image


def load_bvae(run_name):
    save_dir = os.path.join('bvae', 'saved_runs', run_name, 'checkpoints')
    model_name = os.listdir(save_dir)[0]
    save_path = os.path.join(save_dir, model_name)
    state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
    z_dim = state_dict['decoder.0.weight'].size(1)
    model = BetaVAE(z_dim=z_dim)
    model.load_state_dict(state_dict)
    return model


def make_image_grid(image_tensors, pad=2):
    assert 4 <= image_tensors.ndim <= 5
    if image_tensors.ndim == 4:
        grid = make_grid(image_tensors, nrow=1, padding=pad, pad_value=1)
    else:
        _, n_per_row, *img_shape = image_tensors.size()
        image_tensors = image_tensors.view(-1, *img_shape)
        grid = make_grid(image_tensors, nrow=n_per_row, padding=pad, pad_value=1)
    grid = transforms.to_pil_image(grid)
    return grid


def concat_images(img1, img2, pad=2, vertical=False):
    if vertical:
        assert img1.width == img2.width
        new_img = Image.new(img1.mode, (img1.width, img1.height + pad + img2.height), color=(255, 255, 255))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.height + pad))
    else:
        assert img1.height == img2.height
        new_img = Image.new(img1.mode, (img1.width + pad + img2.width, img1.height), color=(255, 255, 255))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width + pad, 0))
    return new_img
