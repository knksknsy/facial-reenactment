import torch
import cv2
import numpy as np

from utils.transforms import normalize

class Resize(object):
    """Resize images and landmarks to given dimension."""

    def __init__(self, image_size, mask_size):
        self.image_size = image_size
        self.mask_size = mask_size


    def __call__(self, sample):
        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']
        mask_real1, mask_real2, mask_fake = sample['mask_real1'], sample['mask_real2'], sample['mask_fake']
        
        image_real1 = cv2.resize(image_real1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_real2 = cv2.resize(image_real2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_fake = cv2.resize(image_fake, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_real1 = cv2.resize(mask_real1, (self.mask_size, self.mask_size), interpolation=cv2.INTER_LINEAR)
        mask_real2 = cv2.resize(mask_real2, (self.mask_size, self.mask_size), interpolation=cv2.INTER_LINEAR)
        mask_fake = cv2.resize(mask_fake, (self.mask_size, self.mask_size), interpolation=cv2.INTER_LINEAR)

        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': mask_real1, 'mask_real2': mask_real2, 'mask_fake': mask_fake,
            'label_real1': sample['label_real1'], 'label_real2': sample['label_real2'], 'label_fake': sample['label_fake']
        }


class GrayScale(object):
    """Convert RGB tensor to grayscale"""

    def __call__(self, sample):
        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']

        image_real1 = cv2.cvtColor(image_real1, cv2.COLOR_BGR2GRAY)
        image_real2 = cv2.cvtColor(image_real2, cv2.COLOR_BGR2GRAY)
        image_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': sample['mask_real1'], 'mask_real2': sample['mask_real2'], 'mask_fake': sample['mask_fake'],
            'label_real1': sample['label_real1'], 'label_real2': sample['label_real2'], 'label_fake': sample['label_fake']
        }


class RandomHorizontalFlip(object):
    """Flip images and landmarks randomly."""

    def __call__(self, sample):
        flip = np.random.rand(1)[0] > 0.5
        if not flip:
            return sample

        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']
        mask_fake = sample['mask_fake']
            
        image_real1 = cv2.flip(image_real1, flipCode=1)
        image_real2 = cv2.flip(image_real2, flipCode=1)
        image_fake = cv2.flip(image_fake, flipCode=1)
        mask_fake = cv2.flip(mask_fake, flipCode=1)

        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': sample['mask_real1'], 'mask_real2': sample['mask_real2'], 'mask_fake': mask_fake,
            'label_real1': sample['label_real1'], 'label_real2': sample['label_real2'], 'label_fake': sample['label_fake']
        }


class RandomRotate(object):
    """Rotate images and landmarks randomly."""

    def __init__(self, angle):
        self.angle = angle


    def __call__(self, sample):
        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']
        mask_fake = sample['mask_fake']

        angle_tmp = np.clip(np.random.rand(1) * self.angle, -40.0, 40.0)
        image_real1 = self.affine_transform(image_real1, np.clip(np.random.rand(1) * self.angle, -40.0, 40.0))
        image_real2 = self.affine_transform(image_real2, np.clip(np.random.rand(1) * self.angle, -40.0, 40.0))
        image_fake = self.affine_transform(image_fake, angle_tmp)
        mask_fake = self.affine_transform(mask_fake, angle_tmp)
        
        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': sample['mask_real1'], 'mask_real2': sample['mask_real2'], 'mask_fake': mask_fake,
            'label_real1': sample['label_real1'], 'label_real2': sample['label_real2'], 'label_fake': sample['label_fake']
        }


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
        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']
        label_real1, label_real2, label_fake = sample['label_real1'], sample['label_real2'], sample['label_fake']
        mask_real1, mask_real2, mask_fake = sample['mask_real1'], sample['mask_real2'], sample['mask_fake']

        if self.channels == 1:
            image_real1, image_real2, image_fake = image_real1[:,:,None], image_real2[:,:,None], image_fake[:,:,None]
            
        mask_real1, mask_real2, mask_fake = mask_real1[:,:,None], mask_real2[:,:,None],  mask_fake[:,:,None]

        # Convert BGR to RGB
        image_real1 = np.ascontiguousarray(image_real1.transpose(2, 0, 1).astype(np.float32))
        image_real2 = np.ascontiguousarray(image_real2.transpose(2, 0, 1).astype(np.float32))
        image_fake = np.ascontiguousarray(image_fake.transpose(2, 0, 1).astype(np.float32))
        mask_real1 = np.ascontiguousarray(mask_real1.transpose(2, 0, 1).astype(np.float32))
        mask_real2 = np.ascontiguousarray(mask_real2.transpose(2, 0, 1).astype(np.float32))
        mask_fake = np.ascontiguousarray(mask_fake.transpose(2, 0, 1).astype(np.float32))

        label_real1 = np.ascontiguousarray(label_real1).astype(np.float32)
        label_real2 = np.ascontiguousarray(label_real2).astype(np.float32)
        label_fake = np.ascontiguousarray(label_fake).astype(np.float32)
        
        # Convert to Tensor
        image_real1 = torch.from_numpy(image_real1 * (1.0 / 255.0)).to(self.device)
        image_real2 = torch.from_numpy(image_real2 * (1.0 / 255.0)).to(self.device)
        image_fake = torch.from_numpy(image_fake * (1.0 / 255.0)).to(self.device)
        mask_real1 = torch.from_numpy(mask_real1 * (1.0 / 255.0)).to(self.device)
        mask_real2 = torch.from_numpy(mask_real2 * (1.0 / 255.0)).to(self.device)
        mask_fake = torch.from_numpy(mask_fake * (1.0 / 255.0)).to(self.device)
        
        label_real1 = torch.from_numpy(label_real1).to(self.device)
        label_real2 = torch.from_numpy(label_real2).to(self.device)
        label_fake = torch.from_numpy(label_fake).to(self.device)

        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': mask_real1, 'mask_real2': mask_real2, 'mask_fake': mask_fake,
            'label_real1': label_real1, 'label_real2': label_real2, 'label_fake': label_fake
        }


class Normalize(object):

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std


    def __call__(self, sample):
        image_real1, image_real2, image_fake = sample['image_real1'], sample['image_real2'], sample['image_fake']
        mask_real1, mask_real2, mask_fake = sample['mask_real1'], sample['mask_real2'], sample['mask_fake']

        image_real1 = normalize(image_real1, self.mean, self.std)
        image_real2 = normalize(image_real2, self.mean, self.std)
        image_fake = normalize(image_fake, self.mean, self.std)
        mask_real1 = normalize(mask_real1, self.mean, self.std)
        mask_real2 = normalize(mask_real2, self.mean, self.std)
        mask_fake = normalize(mask_fake, self.mean, self.std)

        return {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': mask_real1, 'mask_real2': mask_real2, 'mask_fake': mask_fake,
            'label_real1': sample['label_real1'], 'label_real2': sample['label_real2'], 'label_fake': sample['label_fake']
        }
