from argparse import ArgumentParser
import random
import os
from bvae.trainer import train
from bvae.model import BetaVAE
from bvae.dataset import ImageFilesDataset, ImageFolderDataset
import utils

random.seed(27)


if __name__ == '__main__':
    parser = ArgumentParser(description='Beta-VAE training on a folder of images')
    parser.add_argument('--run_name', required=True, type=str, help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', required=True, type=str, help='path to data folder')
    parser.add_argument('--n_iterations', type=float, default=1e5, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=4, help='beta parameter for B-VAE objective')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--train_test_ratio', type=float, default=0.9,
                        help='ratio of train to test data for splitting (if no train/test folders in data_dir)')
    parser.add_argument('--grayscale', action='store_true', help='whether or not to use grayscale images')
    parser.add_argument('--early_stopping', type=int, default=10, help='stop training early if test loss converges '
                                                                       'over past n epochs (-1 for no early stopping)')
    args = parser.parse_args()
    early_stopping = None if args.early_stopping == -1 else args.early_stopping

    if os.path.exists(os.path.join(args.data_dir, 'train')):
        train_set = ImageFolderDataset(os.path.join(args.data_dir, 'train'), grayscale=args.grayscale, training=True)
        test_set = ImageFolderDataset(os.path.join(args.data_dir, 'test'), grayscale=args.grayscale, training=False)
    else:
        files = utils.recursive_folder_image_paths(args.data_dir)
        random.shuffle(files)
        train_files = files[:int(args.train_test_ratio * len(files))]
        test_files = files[int(args.train_test_ratio * len(files)):]
        train_set = ImageFilesDataset(train_files, grayscale=args.grayscale, training=True)
        test_set = ImageFilesDataset(test_files, grayscale=args.grayscale, training=False)

    model = BetaVAE(z_dim=args.z_dim, nc=train_set.nc)

    train(args.run_name, train_set, test_set,
          args.n_iterations, args.batch_size, args.lr, early_stopping,
          args.beta, args.z_dim)
