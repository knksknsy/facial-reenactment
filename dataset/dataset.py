import os
import torch
import cv2
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch._C import Value

from torch.utils.data import Dataset

class VoxCelebDataset(Dataset):
    """Dataset object for accessing and pre-processing VoxCeleb2 dataset"""

    def __init__(self, dataset_path, csv_file, image_size, channels, landmark_type, shuffle_frames=False, transform=None, train_format=False):
        """
        Instantiate the Dataset.

        :param dataset_path: Path to the folder where the pre-processed dataset is stored.
        :param csv_file: CSV file containing the triplet data-pairs.
        :param shuffle_frames: If True, randomly select a triplet data-pair from the CSV row.
        :param transform: Transformations to be done to triplet data-pairs.
        """
        self.dataset_path = dataset_path
        self.data_frame = pd.read_csv(csv_file)
        self.image_size = image_size
        self.channels = channels
        self.landmark_type = landmark_type
        self.shuffle_frames = shuffle_frames
        self.transform = transform
        self.train_format = train_format

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # CSV structure:
        # 0             2           4           6           8           10          12          14          16
        # landmark1,    landmark2,  landmark3,  landmark4,  landmark5,  landmark6,  landmark7,  landmark8,  landmark9

        # 1             3           5           7           9           11          13          15          17
        # image1,       image2,     image3,     image4,     image5,     image6,     image7,     image8,     image9
        
        # 18            19
        # id,           id_video

        image_cols = [1,3,5,7,9,11,13,15,17]

        identity = self.data_frame.iloc[idx, -2]
        id_video = self.data_frame.iloc[idx, -1]
        path = os.path.join(self.dataset_path, identity, id_video)

        # Training
        if self.train_format:
            if self.shuffle_frames:
                samples_n = 3
                image_col_samples = np.random.choice(image_cols, size=samples_n)

                image1_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[0]])
                image2_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[1]])
                image3_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[2]])

                landmark1_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[0] - 1])
                landmark2_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[1] - 1])
                landmark3_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[2] - 1])
            else:
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
            landmark1 = plot_landmarks(np.load(landmark1_path), self.image_size, image1.shape[0], self.channels, self.landmark_type)
            landmark2 = plot_landmarks(np.load(landmark2_path), self.image_size, image2.shape[0], self.channels, self.landmark_type)
            landmark3 = plot_landmarks(np.load(landmark3_path), self.image_size, image3.shape[0], self.channels, self.landmark_type)

            sample = {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        # Testing
        else:
            if self.shuffle_frames:
                samples_n = 2
                image_col_samples = np.random.choice(image_cols, size=samples_n)

                image1_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[0]])
                image2_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[1]])
                landmark2_path = os.path.join(path, self.data_frame.iloc[idx, image_col_samples[1] - 1])
            else:
                image1_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[0]])
                image2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1]])
                landmark2_path = os.path.join(path, self.data_frame.iloc[idx, image_cols[1] - 1])
            
            image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
            image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
            landmark2 = plot_landmarks(np.load(landmark2_path), self.image_size, image2.shape[0], self.channels, self.landmark_type)

            sample = {'image1': image1, 'image2': image2, 'landmark2': landmark2}

        if self.transform:
            sample = self.transform(sample)

        return sample


def plot_landmarks(landmarks, output_res, input_res, channels, landmark_type):
    ratio = input_res / output_res
    landmarks = landmarks/ratio
    dpi = 100
    fig = plt.figure(figsize=(output_res / dpi, output_res / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.zeros((output_res, output_res, channels)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if landmark_type == 'boundary':
        marker_size = 3*72./dpi/ratio
        # Head
        ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'green')
        # Eyebrows
        ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'orange')
        ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'orange')
        # Nose
        ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'blue')
        ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'blue')
        # Eyes
        ax.fill(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'red', fill=False)
        ax.fill(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'red', fill=False)
        # Mouth
        ax.fill(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'purple', fill=False)
        # Inner-Mouth
        ax.fill(landmarks[60:68, 0], landmarks[60:68, 1], linestyle='-', lw=marker_size, color='white' if channels == 1 else 'purple', fill=False)
    elif landmark_type == 'keypoint':
        marker_size = (4*72./dpi/ratio)**2
        # Head
        ax.scatter(landmarks[0:17, 0], landmarks[0:17, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'green')
        # Eyebrows
        ax.scatter(landmarks[17:22, 0], landmarks[17:22, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'orange')
        ax.scatter(landmarks[22:27, 0], landmarks[22:27, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'orange')
        # Nose
        ax.scatter(landmarks[27:31, 0], landmarks[27:31, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'blue')
        ax.scatter(landmarks[31:36, 0], landmarks[31:36, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'blue')
        # Eyes
        ax.scatter(landmarks[36:42, 0], landmarks[36:42, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'red')
        ax.scatter(landmarks[42:48, 0], landmarks[42:48, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'red')
        # Mouth
        ax.scatter(landmarks[48:60, 0], landmarks[48:60, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'purple')
        # Inner-Mouth
        ax.scatter(landmarks[60:68, 0], landmarks[60:68, 1], marker='o', s=marker_size, color='white' if channels == 1 else 'purple')
    else:
        raise ValueError(f'Wrong type provided: {type}')

    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    canvas_shape = fig.canvas.get_width_height()
    data = np.reshape(buffer, (canvas_shape[0], canvas_shape[1], 3))
    plt.close(fig)

    return data
