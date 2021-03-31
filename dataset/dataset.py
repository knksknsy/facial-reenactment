import os
import torch
import cv2

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from dataset.utils import plot_landmarks

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
