from argparse import ArgumentParser
import shutil
import os
import random
import torch
from bvae.model import BetaVAE
from bvae.dataset import ImageFilesDataset
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(27)


def interp_latent(model, image, z_range, z_intervals):
    if z_intervals % 2 == 0:
        z_intervals += 1

    image = image.to(device)
    img_shape = list(image.size())
    with torch.no_grad():
        z = model.encode(image.unsqueeze(dim=0)).squeeze(dim=0)

    z_interps = []
    interp_vals = torch.linspace(start=-z_range, end=z_range, steps=z_intervals).to(device)
    for z_dim in range(model.z_dim):
        for val in interp_vals:
            z_interp = z.clone()
            z_interp[z_dim] += val
            z_interps.append(z_interp)
    z_interps = torch.stack(z_interps)

    with torch.no_grad():
        recon_interp = model.decode(z_interps).view(model.z_dim, z_intervals, *img_shape).cpu()

    return recon_interp


if __name__ == '__main__':
    parser = ArgumentParser(description='Visualize the dimensions of a trained Beta-VAE')
    parser.add_argument('--run_name', required=True, type=str, help='run name of the trained model')
    parser.add_argument('--data_dir', required=True, type=str, help='path to data folder for seed images')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='where to save results (if not specified, saves at bvae/saved_runs/[run_name]/dimensions/)')
    parser.add_argument('--n_seeds', type=int, default=10, help='how many seed images to slide dimensions for')
    parser.add_argument('--z_range', type=float, default=10, help='range in which to slide each dimension')
    parser.add_argument('--z_intervals', type=int, default=10, help='number of intervals in range')
    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = os.path.join('bvae', 'saved_runs', args.run_name, 'dimensions')
    else:
        save_dir = args.save_dir
    shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    files = utils.recursive_folder_image_paths(args.data_dir)
    random.shuffle(files)
    files = files[:args.n_seeds]
    seed_set = ImageFilesDataset(files, training=False)

    model = utils.load_bvae(args.run_name).to(device)

    orig_images = []
    recon_interp_images = []
    for i in range(len(seed_set)):
        image = seed_set[i]
        orig_images.append(image)
        recon_interp_images.append(interp_latent(model, image, args.z_range, args.z_intervals))
    orig_images = torch.stack(orig_images, dim=0)
    recon_interp_images = torch.stack(recon_interp_images, dim=1)

    orig_images = utils.make_image_grid(orig_images)
    for z_dim in range(len(recon_interp_images)):
        dim_interp = recon_interp_images[z_dim]
        dim_interp = utils.make_image_grid(dim_interp)
        dim_interp = utils.concat_images(orig_images, dim_interp, pad=8)
        dim_interp.save('{}/{}.jpg'.format(save_dir, z_dim + 1))
