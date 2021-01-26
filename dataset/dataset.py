import os
import random
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from cv2 import cv2

class VoxCelebDataset(Dataset):
    """Dataset object for accessing and pre-processing VoxCeleb2 dataset"""

    def __init__(self, dataset_path, csv_file, shuffle_frames=False, transform=None):
        """
        Instantiate the Dataset.

        :param dataset_path: Path to the folder where the pre-processed dataset is stored.
        :param csv_file: CSV file containing the triplet data-pairs.
        :param shuffle_frames: If True, randomly select a triplet data-pair from the CSV row.
        :param transform: Transformations to be done to triplet data-pairs.
        """
        self.dataset_path = dataset_path
        self.data_frame = pd.read_csv(csv_file)
        self.shuffle_frames = shuffle_frames
        self.transform = transform

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

        identity = self.data_frame.iloc[idx, 18]
        id_video = self.data_frame.iloc[idx, 19]
        path = os.path.join(self.dataset_path, identity, id_video)

        if self.shuffle_frames:
            samples_n = 3
            image_col_samples = random.choices(image_cols, k=samples_n)

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
        landmark1 = plot_landmarks(image1, np.load(landmark1_path))
        landmark2 = plot_landmarks(image2, np.load(landmark2_path))
        landmark3 = plot_landmarks(image3, np.load(landmark3_path))

        sample = {'image1': image1, 'image2': image2, 'image3': image3, 'landmark1': landmark1, 'landmark2': landmark2, 'landmark3': landmark3}

        if self.transform:
            sample = self.transform(sample)

        return sample


def plot_landmarks(frame, landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a OpenCV Image.
    """
    # TODO: plot landmarks using cv2 (better performance?)
    dpi = 100
    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.zeros(frame.shape))
    # plt.imshow(frame)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)
    # TODO: Inner-Mouth
    ax.plot(landmarks[60:68, 0], landmarks[60:68, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = cv2.imdecode(np.frombuffer(fig.canvas.tostring_rgb(), np.uint8), cv2.IMREAD_COLOR)
    plt.close(fig)
    return data
