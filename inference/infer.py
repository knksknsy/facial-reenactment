import torch
import cv2
import numpy as np

from datetime import datetime
from torchvision import transforms
from face_alignment import FaceAlignment, LandmarksType

from dataset.dataset import plot_landmarks
from loggings.logger import Logger
from configs.options import Options
from models.network import Network

class Infer():
    def __init__(self, logger: Logger, options: Options, model_path: str):
        self.logger = logger
        self.options = options
        self.model_path = model_path
        self.network = Network(self.logger, self.options, self.model_path)
        self.network.eval()


    # TODO: outsource face-alignment code
    def from_image(self):
        self.logger.log_info(f'Source image: {self.options.source}')
        self.logger.log_info(f'Target image: {self.options.target}')

        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)

        source, bbox_s = self._get_image_and_bbox(self.options.source, fa)
        target, bbox_t = self._get_image_and_bbox(self.options.target, fa)

        source = self._crop_and_resize(source, bbox_s, padding=20)
        target = self._crop_and_resize(target, bbox_t, padding=20)

        self.logger.log_info('Extracting facial landmarks from target image.')
        target_landmarks = fa.get_landmarks_from_image(target)[0]
        target_landmarks = plot_landmarks(target, target_landmarks)

        normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        source = torch.FloatTensor(np.ascontiguousarray(source.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))).to(self.options.device)
        target_landmarks = torch.FloatTensor(np.ascontiguousarray(target_landmarks.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))).to(self.options.device)
        source = normalize(source)
        target_landmarks = normalize(target_landmarks)

        output, output_mask, output_color =  self.network(source, target_landmarks)
        self.logger.save_image(self.options.output_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', output, nrow=self.options.batch_size)


    # TODO: implement video inference
    def from_video(self):
        pass


    def _get_image_and_bbox(self, path, face_alignment):
        self.logger.log_info('Extracting bounding boxes from source and target images.')

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        bboxes = face_alignment.face_detector.detect_from_image(image)
        assert len(bboxes) != 0, f'No face detected in {path}'
        
        return image, bboxes[0]


    def _crop_and_resize(self, image, bbox, padding):
        self.logger.log_info('Cropping faces and resizing source and target images.')
        height, width, _ = image.shape
        bbox_x1, bbox_x2 = bbox[0], bbox[2]
        bbox_y1, bbox_y2 = bbox[1], bbox[3]
        
        out_of_bounds = bbox_x1 < padding or bbox_y1 < padding or bbox_x2 >= width - padding or bbox_y2 >= height - padding
        if out_of_bounds:
            image = np.pad(image, padding)
        image = image[bbox_y1 - padding: bbox_y2 + padding, bbox_x1 - padding: bbox_x2 + padding]
        image = cv2.resize(image, (self.options.image_size, self.options.image_size), interpolation=cv2.INTER_LINEAR)

        return image
