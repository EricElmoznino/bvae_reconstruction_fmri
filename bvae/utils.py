import os
import math
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import functional as transforms


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
