from argparse import ArgumentParser
import os
import random
import numpy as np
from tqdm import tqdm
import torch
from fmri.fmri_decoder import fMRIDecoder
from fmri.decoder_training.regression import cv_regression
from fmri.decoder_training.plotting import plot_r
import utils

random.seed(27)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_condition_latents(model, data_dir):
    print('Extracting features')
    stimuli = utils.listdir(data_dir, path=False)
    condition_features = {}
    batch_size = 256
    for i in tqdm(range(0, len(stimuli), batch_size)):
        batch_names = stimuli[i:i + batch_size]
        batch = [utils.load_image(os.path.join(data_dir, n)) for n in batch_names]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            batch_feats = model.encode(batch).cpu().numpy()
        for name, feats in zip(batch_names, batch_feats):
            condition_features[name] = feats
    return condition_features


def get_condition_voxels(subj_file, rois):
    voxels = np.load(subj_file, allow_pickle=True).item()
    voxels = {c: [v[r] for r in rois] for c, v in voxels.items()}
    voxels = {c: np.concatenate(v) for c, v in voxels.items()}
    voxels = {c: torch.from_numpy(v) for c, v in voxels.items()}
    return voxels


if __name__ == '__main__':
    parser = ArgumentParser(description='Train decoder to predict Beta-VAE latents using bold5000 study fMRI data')
    parser.add_argument('--run_name', required=True, type=str, help='name of the current run (where runs are saved)')
    parser.add_argument('--bvae_name', required=True, type=str, help='name of the saved Beta-VAE run')
    parser.add_argument('--data_dir', required=True, type=str, help='path to bold5000 data folder')
    parser.add_argument('--rois', nargs='+', default=['PPA'], type=str,
                        help='ROIs to fit, separated by spaces. '
                             'Options include: EVC, LOC, PFS, OPA, PPA, RSC, FFA, OFA, STS, EBA')
    parser.add_argument('--l2', default=100000, type=float, help='L2 regularization weight')
    args = parser.parse_args()

    bvae = utils.load_bvae(args.bvae_name)
    latents = get_condition_latents(bvae, os.path.join(args.data_dir, 'stimuli'))
    voxels = get_condition_voxels(os.path.join(args.data_dir, 'subj1.npy'), args.rois)
    conditions = list(voxels.keys())
    random.shuffle(conditions)
    latents = np.stack([latents[c] for c in conditions])
    voxels = np.stack([voxels[c] for c in conditions])

    weights, cv_rs = cv_regression(x=voxels, y=latents, n_splits=5, l2=args.l2)
    print('Mean correlation (r) over cross-validated folds\nTraining: {:.4g}\nTest: {:.4g}'
          .format(np.mean(cv_rs['train']), np.mean(cv_rs['test'])))
    plot_r(cv_rs, os.path.join('fmri/saved_runs', args.run_name + '_r.jpg'))

    fmri_decoder = fMRIDecoder(bvae.decoder, weights)
    torch.save(fmri_decoder, os.path.join('fmri/saved_runs', args.run_name + '.pth'))
