import torch
import cv2
import numpy as np

from dataset.utils import normalize

class Resize(object):
    """Resize images and landmarks to given dimension."""

    def __init__(self, size):
        self.size = size


    def __call__(self, sample):
        image_real, image_fake = sample['image_real'], sample['image_fake']
        
        image_real = cv2.resize(image_real, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        image_fake = cv2.resize(image_fake, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': sample['label_real'], 'label_fake': sample['label_fake']}


class GrayScale(object):
    """Convert RGB tensor to grayscale"""

    def __call__(self, sample):
        image_real, image_fake = sample['image_real'], sample['image_fake']

        image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
        image_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': sample['label_real'], 'label_fake': sample['label_fake']}


class RandomHorizontalFlip(object):
    """Flip images and landmarks randomly."""

    def __call__(self, sample):
        flip = np.random.rand(1)[0] > 0.5
        if not flip:
            return sample

        image_real, image_fake = sample['image_real'], sample['image_fake']
            
        image_real = cv2.flip(image_real, flipCode=1)
        image_fake = cv2.flip(image_fake, flipCode=1)

        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': sample['label_real'], 'label_fake': sample['label_fake']}


class RandomRotate(object):
    """Rotate images and landmarks randomly."""

    def __init__(self, angle):
        self.angle = angle


    def __call__(self, sample):
        image_real, image_fake = sample['image_real'], sample['image_fake']

        angle_tmp = np.clip(np.random.rand(1) * self.angle, -40.0, 40.0)
        image_real = self.affine_transform(image_real, angle_tmp)
        image_fake = self.affine_transform(image_fake, angle_tmp)
        
        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': sample['label_real'], 'label_fake': sample['label_fake']}


    def affine_transform(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle.item(), scale=1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, channels, device):
        self.channels = channels
        self.device = device


    def __call__(self, sample):
        image_real, image_fake = sample['image_real'], sample['image_fake']
        label_real, label_fake = sample['label_real'], sample['label_fake']

        if self.channels == 1:
            image_real, image_fake = image_real[:,:,None], image_fake[:,:,None]

        # Convert BGR to RGB
        image_real = np.ascontiguousarray(image_real.transpose(2, 0, 1).astype(np.float32))
        image_fake = np.ascontiguousarray(image_fake.transpose(2, 0, 1).astype(np.float32))

        label_real = np.ascontiguousarray(label_real).astype(np.float32)
        label_fake = np.ascontiguousarray(label_fake).astype(np.float32)
        
        # Convert to Tensor
        image_real = torch.from_numpy(image_real * (1.0 / 255.0)).to(self.device)
        image_fake = torch.from_numpy(image_fake * (1.0 / 255.0)).to(self.device)
        
        label_real = torch.from_numpy(label_real).to(self.device)
        label_fake = torch.from_numpy(label_fake).to(self.device)

        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': label_real, 'label_fake': label_fake}


class Normalize(object):

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std


    def __call__(self, sample):
        image_real, image_fake = sample['image_real'], sample['image_fake']

        image_real = normalize(image_real, self.mean, self.std)
        image_fake = normalize(image_fake, self.mean, self.std)
        
        return {'image_real': image_real, 'image_fake': image_fake, 'label_real': sample['label_real'], 'label_fake': sample['label_fake']}
