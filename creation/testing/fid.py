import torch
import numpy as np
from scipy import linalg
import torch.nn.functional as F

from configs.options import Options
from ..testing.inceptionv3 import InceptionNetwork

class FrechetInceptionDistance():
    def __init__(self, options: Options, device: str, data_loader_length, batch_size):
        self.options = options
        self.device = device
        self.batch_size = batch_size
        self.data_loader_length = data_loader_length * batch_size

        self.inception_network = InceptionNetwork(self.options)
        self.inception_network.to(device)
        self.inception_network.eval()

        self.inception_activations_real = np.zeros((self.data_loader_length, 2048), dtype=np.float32)
        self.inception_activations_fake = np.zeros((self.data_loader_length, 2048), dtype=np.float32)


    def calculate_activations(self, images_real, images_fake, batch_num):
        # Resize images to 299x299
        images_real = F.interpolate(images_real, size=299)
        images_fake = F.interpolate(images_fake, size=299)
        start_idx = self.batch_size * batch_num if batch_num > 1 else 0
        end_idx = self.batch_size * (batch_num + 1) if batch_num > 1 else self.batch_size

        with torch.no_grad():
            activations_real = self.inception_network(images_real).detach().cpu().numpy()
            activations_fake = self.inception_network(images_fake).detach().cpu().numpy()

        if batch_num > 1 and (end_idx - start_idx != activations_real.shape[0]):
            end_idx = start_idx + activations_real.shape[0]

        assert activations_real.shape == (images_real.shape[0], 2048), f'Expexted output shape to be: ({images_real.shape[0]}, 2048), but was: {activations_real.shape}'
        assert activations_fake.shape == (images_fake.shape[0], 2048), f'Expexted output shape to be: ({images_fake.shape[0]}, 2048), but was: {activations_fake.shape}'

        self.inception_activations_real[start_idx:end_idx, :] = activations_real
        self.inception_activations_fake[start_idx:end_idx, :] = activations_fake


    def _calculate_activation_statistics(self, real=True):
        if real:
            activations = self.inception_activations_real
        else:
            activations = self.inception_activations_fake

        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        
        return mu, sigma

    def _calculate_frechet_distance(self, mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
                
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                inception net ( like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """
        
        mu_real = np.atleast_1d(mu_real)
        mu_fake = np.atleast_1d(mu_fake)

        sigma_real = np.atleast_2d(sigma_real)
        sigma_fake = np.atleast_2d(sigma_fake)

        assert mu_real.shape == mu_fake.shape, 'Training and test mean vectors have different lengths'
        assert sigma_real.shape == sigma_fake.shape, 'Training and test covariances have different dimensions'

        diff = mu_real - mu_fake

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-6):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean


    def calculate_fid(self):
        mu_real, sigma_real = self._calculate_activation_statistics(real=True)
        mu_fake, sigma_fake = self._calculate_activation_statistics(real=False)

        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid
