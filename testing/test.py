import os
import logging
import cv2
import torch
import numpy as np
from face_alignment import FaceAlignment, LandmarksType

from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from datetime import datetime

from configs.options import Options
from testing.fid import FrechetInceptionDistance
from dataset import VoxCelebDataset
from dataset import Resize, ToTensor
from dataset import plot_landmarks
from models import Network, save_image

class Test():
    def __init__(self, options: Options, network: Network=None, training=False):
        self.options = options
        self.training = training

        if self.training:
            self.network = network
            self.data_loader_train = self._get_data_loader()
        else:
            self.network = Network(self.options, self.training)


    def _get_data_loader(self):
        dataset_test = VoxCelebDataset(
            dataset_path=self.options.dataset_test,
            csv_file=self.options.csv_test,
            shuffle_frames=self.options.shuffle_frames,
            transform=transforms.Compose([
                        Resize(size=self.options.image_size_test),
                        ToTensor(device=self.options.device)
            ]),
            training=self.training
        )

        data_loader_test = DataLoader(dataset_test,
                                        batch_size=self.options.batch_size,
                                        shuffle=self.options.shuffle,
                                        num_workers=self.options.num_workers,
                                        pin_memory=self.options.pin_memory
        )

        return data_loader_test


    def __call__(self, epoch=None):
        self.network.eval()
        run_start = datetime.now()

        logging.info('===== TESTING =====')
        logging.info(f'Running on {self.options.device.upper()}.')

        if self.training and epoch is not None:
            self._test(epoch)

        elif not self.training and epoch is None:
            self._test_single()

        run_end = datetime.now()
        logging.info(f'Testing finished in {run_end - run_start}.')


    def _test_single(self):
        logging.info(f'Source image: {self.options.source}')
        logging.info(f'Target image: {self.options.target}')

        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)

        source, bbox_s = self._get_image_and_bbox(self.options.source, fa)
        target, bbox_t = self._get_image_and_bbox(self.options.target, fa)

        source = self._crop_and_resize(source, bbox_s, padding=20)
        target = self._crop_and_resize(target, bbox_t, padding=20)

        logging.info('Extracting facial landmarks from target image.')
        target_landmarks = fa.get_landmarks_from_image(target)[0]
        target_landmarks = plot_landmarks(target, target_landmarks)

        normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        source = torch.FloatTensor(np.ascontiguousarray(source.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))).to(self.options.device)
        target_landmarks = torch.FloatTensor(np.ascontiguousarray(target_landmarks.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))).to(self.options.device)
        source = normalize(source)
        target_landmarks = normalize(target_landmarks)

        output =  self.network(source, target_landmarks)
        save_image(self.options.output, f'{datetime.now():%Y%m%d_%H%M%S%f}.png', output)


    def _get_image_and_bbox(self, path, face_alignment):
        logging.info('Extracting bounding boxes from source and target images.')

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        bboxes = face_alignment.face_detector.detect_from_image(image)
        assert len(bboxes) != 0, f'No face detected in {path}'
        
        return image, bboxes[0]


    def _crop_and_resize(self, image, bbox, padding):
        logging.info('Cropping faces and resizing source and target images.')
        height, width, _ = image.shape
        bbox_x1, bbox_x2 = bbox[0], bbox[2]
        bbox_y1, bbox_y2 = bbox[1], bbox[3]
        
        out_of_bounds = bbox_x1 < padding or bbox_y1 < padding or bbox_x2 >= width - padding or bbox_y2 >= height - padding
        if out_of_bounds:
            image = np.pad(image, padding)
        image = image[bbox_y1 - padding: bbox_y2 + padding, bbox_x1 - padding: bbox_x2 + padding]
        image = cv2.resize(image, (self.options.image_size, self.options.image_size), interpolation=cv2.INTER_LINEAR)

        return image


    def _test(self, epoch=None):
        logging.info(f'Batches/Iterations: {len(self.data_loader_test)} Batch Size: {self.options.batch_size}')

        fid = FrechetInceptionDistance(self.options, len(self.data_loader_train))

        # Path for generated test images
        root = os.path.join(os.path.split(self.options.dataset_test)[0], self.options.gen_test_dir)
        if not os.path.isdir(root):
            os.makedirs(root)

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images = batch['image2'].to(self.options.device)
            gen_images = self.network(batch['image1'], batch['landmark2']).to(self.options.device)

            # Save generated images
            for idx in range(len(batch)):
                self.save_image(os.path.join(root, batch['id'][idx], batch['id_video']), 'fake.png', gen_images[idx], epoch, batch_num)

            # Calculate FID
            fid.calculate_activations(images, gen_images, batch_num)

            # Calculate SSIM
            ssim_val = ssim(gen_images, batch['image2'], data_range=255, size_average=False)

            batch_end = datetime.now()
        
        fid_score = fid.calculate_fid()
