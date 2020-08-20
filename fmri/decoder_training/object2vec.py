from argparse import ArgumentParser
import os
import numpy as np
import scipy.io
from tqdm import tqdm
import torch
from fmri.decoder_training.fmri_decoder import fMRIDecoder
from fmri.decoder_training.regression import cv_regression
from fmri.decoder_training.plotting import plot_r
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def mean_condition_latents(model, data_dir):
    print('Extracting stimuli features')
    conditions = utils.listdir(data_dir, path=False)
    condition_features = {}
    for c in tqdm(conditions):
        stimuli = utils.listdir(os.path.join(data_dir, c))
        stimuli = [utils.load_image(s) for s in stimuli]
        stimuli = torch.stack(stimuli).to(device)
        with torch.no_grad():
            feats = model.encode(stimuli).mean(dim=0).cpu().numpy()
        condition_features[c] = feats
    return condition_features


def get_condition_voxels(data_dir, rois, subject_num=1):
        roistack = scipy.io.loadmat('{}/subj{:03}'.format(data_dir, subject_num) + '/roistack.mat')['roistack']
        roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]
        voxels = roistack['betas'][0, 0]
        roi_indices = roistack['indices'][0, 0][0]
        roi_masks = {r: roi_indices == (i + 1) for i, r in enumerate(roi_names)}
        condition_voxels = {cond: np.concatenate([voxels[i][roi_masks[r]] for r in rois])
                            for i, cond in enumerate(conditions)}

        sets = scipy.io.loadmat('{}/subj{:03}'.format(data_dir, subject_num) + '/sets.mat')['sets']
        cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]

        return condition_voxels, cv_sets


if __name__ == '__main__':
    parser = ArgumentParser(description='Train decoder to predict Beta-VAE latents using object2vec study fMRI data')
    parser.add_argument('--run_name', required=True, type=str, help='name of the current run (where runs are saved)')
    parser.add_argument('--bvae_name', required=True, type=str, help='name of the saved Beta-VAE run')
    parser.add_argument('--data_dir', required=True, type=str, help='path to object2vec data folder')
    parser.add_argument('--rois', nargs='+', default=['LOC'], type=str,
                        help='ROIs to fit, separated by spaces. '
                             'Options include: EVC, LOC, PFS, OPA, PPA, RSC, FFA, OFA, STS, EBA')
    parser.add_argument('--l2', default=10, type=float, help='L2 regularization weight')
    args = parser.parse_args()

    bvae = utils.load_bvae(args.bvae_name)
    latents = mean_condition_latents(bvae, os.path.join(args.data_dir, 'stimuli'))
    voxels, cv_sets = get_condition_voxels(args.data_dir, args.rois)
    latents = np.concatenate([np.stack([latents[c] for c in s]) for s in cv_sets])
    voxels = np.concatenate([np.stack([voxels[c] for c in s]) for s in cv_sets])

    weights, cv_rs = cv_regression(x=voxels, y=latents, n_splits=len(cv_sets), l2=args.l2)
    print('Mean correlation (r) over cross-validated folds\nTraining: {:.4g}\nTest: {:.4g}'
          .format(np.mean(cv_rs['train']), np.mean(cv_rs['test'])))
    plot_r(cv_rs, os.path.join('fmri/saved_runs', args.run_name + '_r.jpg'))

    fmri_decoder = fMRIDecoder(bvae.decoder, weights)
    torch.save(fmri_decoder, os.path.join('fmri/decoder_training/saved_runs', args.run_name + '.pth'))
