from argparse import ArgumentParser
from bvae.trainer import train
from bvae.model import BetaVAE
from bvae.dataset import NumpyImageDataset, DebugDataset


if __name__ == '__main__':
    parser = ArgumentParser(description='Beta-VAE training on DSprite dataset from deepmind')
    parser.add_argument('--run_name', required=True, type=str, help='name of the current run (where runs are saved)')
    parser.add_argument('--data_path', required=True, type=str, help='path to data .npz file')
    parser.add_argument('--n_iterations', type=int, default=1.5e6, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=4, help='learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='learning rate')
    args = parser.parse_args()

    train_set = NumpyImageDataset(args.data_path)
    test_set = NumpyImageDataset(args.data_path)
    test_set.data = test_set.data[:max(36, args.batch_size)]    # Mock test set just so that we can use the trainer code

    model = BetaVAE(z_dim=args.z_dim, nc=1, decoder_distribution='bernoulli')

    train(args.run_name, train_set, test_set, args.n_iterations, args.batch_size, args.lr, args.beta, args.z_dim)
