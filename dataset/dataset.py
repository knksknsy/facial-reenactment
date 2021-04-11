import re
import os
import torch
import cv2

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from dataset.utils import plot_landmarks, plot_mask

class VoxCelebDataset(Dataset):
    """Dataset object for accessing and pre-processing VoxCeleb2 dataset"""

    def __init__(self, dataset_path, csv_file, image_size, channels, landmark_type, transform=None, train_format=False):
        """
        Instantiate the Dataset.

        :param dataset_path: Path to the folder where the pre-processed dataset is stored.
        :param csv_file: CSV file containing the triplet data-pairs.
        :param transform: Transformations to be done to triplet data-pairs.
        """
        self.dataset_path = dataset_path
        self.data_frame = pd.read_csv(csv_file)
        self.image_size = image_size
        self.channels = channels
        self.landmark_type = landmark_type
        self.transform = transform
        self.train_format = train_format

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # CSV structure:
        # 0             2           4
        # landmark1,    landmark2,  landmark3

        # 1             3           5
        # image1,       image2,     image3
        
        # 6             7
        # id,           id_video

        image_cols = [1,3,5]

        identity = self.data_frame.iloc[idx, -2]
        id_video = self.data_frame.iloc[idx, -1]
        path = os.path.join(self.dataset_path, identity, id_video)

        # Training
        if self.train_format:
            image1_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[0]])
            image2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1]])
            image3_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[2]])

            landmark1_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[0] - 1])
            landmark2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1] - 1])
            landmark3_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[2] - 1])

            # Read images
            image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
            image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
            image3 = cv2.imread(image3_path, cv2.IMREAD_COLOR)

            # Read landmarks
            landmark1 = plot_landmarks(landmarks=np.load(landmark1_path), output_res=self.image_size, input_res=image1.shape[0], channels=self.channels, landmark_type=self.landmark_type)
            landmark2 = plot_landmarks(landmarks=np.load(landmark2_path), output_res=self.image_size, input_res=image2.shape[0], channels=self.channels, landmark_type=self.landmark_type)
            landmark3 = plot_landmarks(landmarks=np.load(landmark3_path), output_res=self.image_size, input_res=image3.shape[0], channels=self.channels, landmark_type=self.landmark_type)

            sample = {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        # Testing
        else:
            image1_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[0]])
            image2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1]])
            landmark2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1] - 1])
            
            image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
            image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
            landmark2 = plot_landmarks(landmarks=np.load(landmark2_path), output_res=self.image_size, input_res=image2.shape[0], channels=self.channels, landmark_type=self.landmark_type)

            sample = {'image1': image1, 'image2': image2, 'landmark2': landmark2}

        if self.transform:
            sample = self.transform(sample)

        return sample


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


def get_pair_contrastive(batch, real_pair: bool, device: str):
    batch_size = batch['image_real1'].shape[0]
    assert batch_size % 2 == 0, 'batch_size must be even'

    if real_pair:
        pair1 = batch['image_real1']
        pair2 = batch['image_real2']
        labels = torch.ones((batch_size, 1)).to(device)
    else:
        pair1 = batch['image_real1']
        pair2 = batch['image_fake']
        labels = torch.zeros((batch_size, 1)).to(device)

    return pair1, pair2, labels


def get_pair_classification(batch):
    image_real, image_fake = batch['image_real1'], batch['image_fake']
    mask_real, mask_fake = batch['mask_real1'], batch['mask_fake']
    label_real, label_fake = batch['label_real1'], batch['label_fake']

    images = torch.cat((image_real, image_fake), dim=0)
    masks = torch.cat((mask_real, mask_fake), dim=0)
    labels = torch.cat((label_real, label_fake), dim=0)

    indexes = torch.randperm(images.shape[0])
    images = images[indexes]
    masks = masks[indexes]
    labels = labels[indexes]

    return images, labels


    # # For triple loss
    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
        
    #     # CSV structure:
    #     # 0               1               2
    #     # image_real,     label_real,     id_real

    #     # 3               4               5
    #     # image_fake,     label_fake,     id_fake

    #     # 6               7
    #     # landmark_fake   method

    #     image_cols = [0, 3]
    #     label_cols = [1, 4]
    #     id_cols = [2, 5]
    #     method = self.data_frame.iloc[idx, -1]

    #     # Anchor
    #     image_anchor_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, image_cols[0]])
    #     image_anchor = cv2.imread(image_anchor_path, cv2.IMREAD_COLOR)
    #     mask_anchor = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
    #     label_anchor = self.data_frame.iloc[idx, label_cols[0]]
    #     id_anchor = self.data_frame.iloc[idx, id_cols[0]]

    #     # Positive
    #     # Randomly select different frame of the same identity (from anchor)
    #     anchor_num = int(re.search(r'_(.*?).png', image_anchor_path.split('/')[-1]).group(1))
    #     idx_choice = list(range(self.num_images_per_identity))
    #     idx_choice.remove(anchor_num)
    #     image_pos_path = '/'.join(image_anchor_path.split('/')[:-1]) + '/' + id_anchor.replace('id', '') + '_' + str(np.random.choice(idx_choice, 1)[0]) + '.png'
    #     image_pos = cv2.imread(image_pos_path, cv2.IMREAD_COLOR)
    #     mask_pos = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
    #     label_pos = self.data_frame.iloc[idx, label_cols[0]]
    #     id_pos = self.data_frame.iloc[idx, id_cols[0]]

    #     # Negative
    #     image_neg_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, image_cols[1]])
    #     image_neg = cv2.imread(image_neg_path, cv2.IMREAD_COLOR)
    #     landmark_neg_path = os.path.join(self.dataset_path, self.data_frame.iloc[idx, -2])
    #     mask_neg = plot_mask(landmarks=np.load(landmark_neg_path), output_res=self.image_size, input_res=image_neg.shape[0])
    #     label_neg = self.data_frame.iloc[idx, label_cols[1]]
    #     id_neg = self.data_frame.iloc[idx, id_cols[1]]

    #     sample = {'anchor': image_anchor, 'positive': image_pos, 'negative': image_neg}
    #     # sample = {'image_real': image_real, 'image_fake': image_fake, 'mask_real': mask_real, 'mask_fake': mask_fake, 'label_real': label_real, 'label_fake': label_fake}

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample
