import os
from numpy.core.fromnumeric import std
import torch
import cv2
import numpy as np

from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from utils.preprocess import crop_frame, extract_frames, get_bounding_box, plot_landmarks
from utils.transforms import normalize

from loggings.logger import Logger
from configs.options import Options

from ..models.network import Network

class Infer():

    def __init__(self, logger: Logger, options: Options, source: str, model_path: str):
        self.logger = logger
        self.options = options
        self.source = source
        self.padding = self.options.padding
        self.model_path = model_path

        self.network = Network(self.logger, self.options, self.model_path)
        self.network.eval()

        self.fa = FaceAlignment(LandmarksType._2D, device=self.options.device)


    def from_image(self, filename: str = None):
        color_fake = (0, 0, 255)
        color_real = (0, 255, 0)

        self.logger.log_info(f'Image: {self.source}')
        source_raw = cv2.imread(self.source, cv2.IMREAD_COLOR)

        self.logger.log_info('Cropping and resizing image...')
        source, rmin, rmax, cmin, cmax = self.detect_crop_face(source_raw, padding=self.padding, face_alignment=self.fa)

        # Pad bounding box
        rmin = np.clip(rmin - 2 * self.padding, 0, source_raw.shape[0])
        rmax = np.clip(rmax, 0, source_raw.shape[0])
        cmin = np.clip(cmin - 2 * self.padding, 0, source_raw.shape[1])
        cmax = np.clip(cmax, 0, source_raw.shape[1])

        if source is None:
            self.logger.log_info('Could not find any faces!')
            return

        self.logger.log_info('Detecting facial reenactment...')

        source = np.ascontiguousarray(source.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
        source = torch.from_numpy(source * (1.0 / 255.0)).to(self.options.device)
        source = normalize(source, mean=self.options.normalize[0], std=self.options.normalize[1])

        output, _ = self.network(source)
        prediction = output.item() > self.options.threshold
        cv2.rectangle(source_raw, (cmin, rmin), (cmax, rmax), color=color_fake if not prediction else color_real, thickness=3)

        if filename is None:
            filename = f't_{datetime.now():%Y%m%d_%H%M%S}.png'
        else:
            filename = f'{filename}_t_{datetime.now():%Y%m%d_%H%M%S}.png'
        cv2.imwrite(os.path.join(self.options.output_dir, filename), source_raw)
        self.logger.log_info(f'Detecting facial reenactment done. Image saved in {filename}.')


    def from_video(self, filename: str = None, output_path: str = None):
        color_fake = (0, 0, 255)
        color_real = (0, 255, 0)

        self.logger.log_info(f'Image: {self.source}')

        self.logger.log_info('Cropping and resizing video...')
        source_frames_raw = np.concatenate([extract_frames(self.source)])
        source_frames_filtered = []
        bounding_boxes = []
        for i, source_frame in enumerate(source_frames_raw):
            f, rmin, rmax, cmin, cmax = self.detect_crop_face(source_frame, self.padding, face_alignment=self.fa)
            if f is not None:
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                source_frames_filtered.append(f)

                # Pad bounding box
                rmin = np.clip(rmin - 2 * self.padding, 0, source_frame.shape[0])
                rmax = np.clip(rmax, 0, source_frame.shape[0])
                cmin = np.clip(cmin - 2 * self.padding, 0, source_frame.shape[1])
                cmax = np.clip(cmax, 0, source_frame.shape[1])
                bounding_boxes.append([(cmin, rmin), (cmax, rmax)])
                print(f'[{i + 1}/{len(source_frames_raw)}] done', end='\r')
            else:
                del source_frames_raw[i]
                # bounding_boxes.append([(0, 0), (source_frame.shape[0], source_frame.shape[1])])

        if len(source_frames_filtered) == 0:
            self.logger.log_info('Could not find any faces in video!')
            return

        self.logger.log_info('Detecting facial reenactment...')
        predictions = []
        for i, source_frame in enumerate(source_frames_filtered):
            source_frame = np.ascontiguousarray(source_frame.transpose(2,0,1)[None, :, :, :].astype(np.float32))
            source_frame = torch.from_numpy(source_frame * (1.0 / 255.0)).to(self.options.device)
            source_frame = normalize(source_frame, mean=self.options.normalize[0], std=self.options.normalize[1])

            output, _ = self.network(source_frame)
            prediction = output.item() > self.options.threshold
            predictions.append(prediction)

        ext = '.mp4'
        if filename is None:
            filename = f't_{datetime.now():%Y%m%d_%H%M%S}{ext}'
        else:
            filename = f'{filename}_t_{datetime.now():%Y%m%d_%H%M%S}{ext}'

        if not os.path.isdir(self.options.output_dir):
            os.makedirs(self.options.output_dir)

        if output_path is None:
            filename = os.path.join(self.options.output_dir, filename)
        else:
            filename = os.path.join(output_path, filename)

        self.logger.log_info('Saving video...')
        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=25.0, frameSize=source_frames_raw[0].shape[:2][::-1])
        for i, (source_frame_raw, bounding_box, prediction) in enumerate(zip(source_frames_raw, bounding_boxes, predictions)):
            source_frame_raw = cv2.cvtColor(source_frame_raw, cv2.COLOR_RGB2BGR)            
            cv2.rectangle(source_frame_raw, bounding_box[0], bounding_box[1], color=color_fake if not prediction else color_real, thickness=3)
            video_writer.write(source_frame_raw)
            print(f'[{i + 1}/{len(predictions)}] done', end='\r')
        video_writer.release()

        self.logger.log_info(f'Detecting facial reenactment done. Video saved in {filename}.')


    def detect_crop_face(self, frame, padding, face_alignment):
        frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        landmarks = face_alignment.get_landmarks_from_image(frame)
        face_detected = landmarks is not None
        if not face_detected:
            return None, None
        else:
            landmarks = landmarks[0]
            frame, rmin, rmax, cmin, cmax = get_bounding_box(frame, landmarks, (self.options.image_size,self.options.image_size), padding, method='cv2')

            if frame is None:
                return None, None
                    
            return frame, rmin, rmax, cmin, cmax
