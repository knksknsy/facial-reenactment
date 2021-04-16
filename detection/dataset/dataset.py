import re
import os
import torch
import cv2

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from utils.preprocess import plot_mask


class FaceForensicsDataset(Dataset):
    """Dataset object for accessing and pre-processing FaceForensics dataset"""

    def __init__(self, num_images_per_identity, dataset_path, csv_file, image_size, channels, transform=None):
        """
        Instantiate the Dataset.

        :param dataset_path: Path to the folder where the pre-processed dataset is stored.
        :param csv_file: CSV file containing the triplet data-pairs.
        :param transform: Transformations to be done to triplet data-pairs.
        """
        self.num_images_per_identity = num_images_per_identity
        self.dataset_path = dataset_path
        self.data_frame = pd.read_csv(csv_file)
        self.image_size = image_size
        self.channels = channels
        self.transform = transform


    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data_frame)


    # For contrastive loss
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # CSV structure:
        # 0                 1               2
        # image_real,       label_real,     id_real

        # 3                 4               5
        # image_fake,       label_fake,     id_fake

        # 6                 7
        # landmark_fake     method

        image_cols = [0, 3]
        label_cols = [1, 4]
        id_cols = [2, 5]
        method = self.data_frame.iloc[idx, -1]

        # Real image
        image_real1_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, image_cols[0]])
        image_real1 = cv2.imread(image_real1_path, cv2.IMREAD_COLOR)
        mask_real1 = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
        label_real1 = self.data_frame.iloc[idx, label_cols[0]]
        id_real1 = self.data_frame.iloc[idx, id_cols[0]]

        # Alternative real image
        # Randomly select different frame of the same identity
        real1_frame_num = int(re.search(r'_(.*?).png', image_real1_path.split('/')[-1]).group(1))
        idx_choice = list(range(self.num_images_per_identity))
        idx_choice.remove(real1_frame_num)
        image_real2_path = '/'.join(image_real1_path.split('/')[:-1]) + '/' + id_real1.replace('id', '') + '_' + str(np.random.choice(idx_choice, 1)[0]) + '.png'
        image_real2 = cv2.imread(image_real2_path, cv2.IMREAD_COLOR)
        mask_real2 = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
        label_real2 = self.data_frame.iloc[idx, label_cols[0]]
        id_real2 = self.data_frame.iloc[idx, id_cols[0]]

        # Fake image
        image_fake_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, image_cols[1]])
        image_fake = cv2.imread(image_fake_path, cv2.IMREAD_COLOR)
        landmark_fake_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, -2])
        mask_fake = plot_mask(landmarks=np.load(landmark_fake_path), output_res=self.image_size, input_res=image_fake.shape[0])
        label_fake = self.data_frame.iloc[idx, label_cols[1]]
        id_fake = self.data_frame.iloc[idx, id_cols[1]]

        sample = {
            'image_real1': image_real1, 'image_real2': image_real2, 'image_fake': image_fake,
            'mask_real1': mask_real1, 'mask_real2': mask_real2, 'mask_fake': mask_fake,
            'label_real1': label_real1, 'label_real2': label_real2, 'label_fake': label_fake
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_pair_feature(sample, real_pair: bool, device: str):
    batch_size = sample['image_real1'].shape[0]

    if real_pair:
        pair1 = sample['image_real1']
        pair2 = sample['image_real2']
        labels = torch.ones((batch_size, 1)).to(device)
        mask1 = sample['mask_real1']
        mask2 = sample['mask_real2']
    else:
        pair1 = sample['image_real1']
        pair2 = sample['image_fake']
        labels = torch.zeros((batch_size, 1)).to(device)
        mask1 = sample['mask_real1']
        mask2 = sample['mask_fake']

    return pair1, pair2, labels, mask1, mask2


def get_pair_classification(sample):
    image_real, image_fake = sample['image_real1'], sample['image_fake']
    mask_real, mask_fake = sample['mask_real1'], sample['mask_fake']
    label_real, label_fake = sample['label_real1'], sample['label_fake']

    images = torch.cat((image_real, image_fake), dim=0)
    masks = torch.cat((mask_real, mask_fake), dim=0)
    labels = torch.cat((label_real, label_fake), dim=0)

    indexes = torch.randperm(images.shape[0])
    images = images[indexes]
    masks = masks[indexes]
    labels = labels[indexes]

    return images, labels, masks


# def get_triplet_feature(sample, real_pair: bool):
#     batch_size = sample['image_real1'].shape[0]
#     assert batch_size % 2 == 0, 'batch_size must be an even integer.'

#     anchor1 = sample['image_real1'][:batch_size//2]
#     positive1 = sample['image_real2'][:batch_size//2]
#     negative1 = sample['image_fake'][:batch_size//2]

#     anchor2 = sample['image_real1'][batch_size//2:]
#     positive2 = sample['image_real2'][batch_size//2:]
#     negative2 = sample['image_fake'][batch_size//2:]

#     pair1 = {'anchor': anchor1, 'positive': positive1, 'negative': negative1}
#     pair2 = {'anchor': anchor2, 'positive': positive2, 'negative': negative2}

#     return pair1, pair2