import torch
import cv2
import numpy as np

class Resize(object):
    """Resize images and landmarks to given dimension."""

    def __init__(self, size, training=True):
        self.size = size
        self.training = training


    def __call__(self, sample):

        if self.training:
            image1, image2, image3 = sample['image1'], sample['image2'], sample['image3']
            landmark1, landmark2, landmark3 = sample['landmark1'], sample['landmark2'], sample['landmark3']
            
            image1 = cv2.resize(image1, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image2 = cv2.resize(image2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image3 = cv2.resize(image3, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            landmark1 = cv2.resize(landmark1, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            landmark2 = cv2.resize(landmark2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            landmark3 = cv2.resize(landmark3, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

            return {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        else:
            image1, image2, landmark2 = sample['image1'], sample['image2'], sample['landmark2']
            
            image1 = cv2.resize(image1, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image2 = cv2.resize(image2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            landmark2 = cv2.resize(landmark2, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

            return {'image1': image1, 'image2': image2, 'image3': None, 'landmark1': None, 'landmark2': landmark2, 'landmark3': None}


class RandomHorizontalFlip(object):
    """Flip images and landmarks randomly."""

    def __init__(self, training=True):
        self.training = training


    def __call__(self, sample):
        flip = np.random.rand(1)[0] > 0.5
        if not flip:
            return sample

        if self.training:
            image1, image2, image3 = sample['image1'], sample['image2'], sample['image3']
            landmark1, landmark2, landmark3 = sample['landmark1'], sample['landmark2'], sample['landmark3']
                
            image1 = cv2.flip(image1, flipCode=1)
            image2 = cv2.flip(image2, flipCode=1)
            image3 = cv2.flip(image3, flipCode=1)
            landmark1 = cv2.flip(landmark1, flipCode=1)
            landmark2 = cv2.flip(landmark2, flipCode=1)
            landmark3 = cv2.flip(landmark3, flipCode=1)

            return {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        else:
            image1, image2, landmark2 = sample['image1'], sample['image2'], sample['landmark2']
                
            image1 = cv2.flip(image1, flipCode=1)
            image2 = cv2.flip(image2, flipCode=1)
            landmark2 = cv2.flip(landmark2, flipCode=1)

            return {'image1': image1, 'image2': image2, 'image3': None, 'landmark1': None, 'landmark2': landmark2, 'landmark3': None}


class RandomRotate(object):
    """Rotate images and landmarks randomly."""

    def __init__(self, angle, training=True):
        self.angle = angle
        self.training = training


    def __call__(self, sample):
        if self.training:
            image1, image2, image3 = sample['image1'], sample['image2'], sample['image3']
            landmark1, landmark2, landmark3 = sample['landmark1'], sample['landmark2'], sample['landmark3']

            # Rotate each image-landmark pair
            for i in range(3):
                angle_tmp = np.clip(np.random.rand(1) * (i + 1) * self.angle, -40.0, 40.0)
                if i == 0:
                    image1 = self.affine_transform(image1, angle_tmp)
                    landmark1 = self.affine_transform(landmark1, angle_tmp)
                if i == 1:
                    image2 = self.affine_transform(image2, angle_tmp)
                    landmark2 = self.affine_transform(landmark2, angle_tmp)
                if i == 2:
                    image3 = self.affine_transform(image3, angle_tmp)
                    landmark3 = self.affine_transform(landmark3, angle_tmp)
            
            return {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        else:
            image1, image2, landmark2 = sample['image1'], sample['image2'], sample['landmark2']
            
            for i in range(2):
                angle_tmp = np.clip(np.random.rand(1) * (i + 1) * self.angle, -40.0, 40.0)
                if i == 0:
                    image1 = self.affine_transform(image1, angle_tmp)
                if i == 1:
                    image2 = self.affine_transform(image2, angle_tmp)
                    landmark2 = self.affine_transform(landmark2, angle_tmp)
            
            return {'image1': image1, 'image2': image2, 'image3': None, 'landmark1': None, 'landmark2': landmark2, 'landmark3': None}


    def affine_transform(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle.item(), scale=1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device, training=True):
        self.device = device
        self.training = training


    def __call__(self, sample):
        if self.training:
            image1, image2, image3 = sample['image1'], sample['image2'], sample['image3']
            landmark1, landmark2, landmark3 = sample['landmark1'], sample['landmark2'], sample['landmark3']

            # Convert BGR to RGB
            image1 = np.ascontiguousarray(image1.transpose(2, 0, 1).astype(np.float32))
            image2 = np.ascontiguousarray(image2.transpose(2, 0, 1).astype(np.float32))
            image3 = np.ascontiguousarray(image3.transpose(2, 0, 1).astype(np.float32))
            landmark1 = np.ascontiguousarray(landmark1.transpose(2, 0, 1).astype(np.float32))
            landmark2 = np.ascontiguousarray(landmark2.transpose(2, 0, 1).astype(np.float32))
            landmark3 = np.ascontiguousarray(landmark3.transpose(2, 0, 1).astype(np.float32))

            # Convert to Tensor
            image1 = torch.FloatTensor(image1).to(self.device)
            image2 = torch.FloatTensor(image2).to(self.device)
            image3 = torch.FloatTensor(image3).to(self.device)
            landmark1 = torch.FloatTensor(landmark1).to(self.device)
            landmark2 = torch.FloatTensor(landmark2).to(self.device)
            landmark3 = torch.FloatTensor(landmark3).to(self.device)

            return {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        else:
            image1, image2, landmark2 = sample['image1'], sample['image2'], sample['landmark2']

            # Convert BGR to RGB
            image1 = np.ascontiguousarray(image1.transpose(2, 0, 1).astype(np.float32))
            image2 = np.ascontiguousarray(image2.transpose(2, 0, 1).astype(np.float32))
            landmark2 = np.ascontiguousarray(landmark2.transpose(2, 0, 1).astype(np.float32))

            # Convert to Tensor
            image1 = torch.FloatTensor(image1).to(self.device)
            image2 = torch.FloatTensor(image2).to(self.device)
            landmark2 = torch.FloatTensor(landmark2).to(self.device)

            return {'image1': image1, 'image2': image2, 'image3': None, 'landmark1': None, 'landmark2': landmark2, 'landmark3': None}
