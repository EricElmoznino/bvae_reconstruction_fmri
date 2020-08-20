import torch
import numpy as np
from torch import nn


class fMRIDecoder(nn.Module):

    def __init__(self, image_decoder, weights):
        super().__init__()

        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)

        self.n_voxels = weights.size(1)
        self.n_latents = weights.size(0)

        self.latent_projection = nn.Linear(self.n_voxels, self.n_latents, bias=False)
        self.latent_projection.weight.data = weights

        self.image_decoder = image_decoder

        self.eval()

    def forward(self, voxels, gen_image=True):
        latent = self.latent_projection(voxels)
        if gen_image:
            image = self.image_decoder(latent)
            return latent, image
        else:
            return latent
