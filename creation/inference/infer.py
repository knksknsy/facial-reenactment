import os
import torch
import cv2
import numpy as np

from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from utils.preprocess import crop_frame, extract_frames, plot_landmarks
from utils.transforms import normalize
from loggings.logger import Logger
from configs.options import Options
from ..models.network import Network

class Infer():
    def __init__(self, logger: Logger, options: Options, source: str, target: str, model_path: str):
        self.logger = logger
        self.options = options
        self.source = source
        self.target = target
        self.model_path = model_path

        self.network = Network(self.logger, self.options, self.model_path)
        self.network.eval()

        self.fa = FaceAlignment(LandmarksType._2D, device=self.options.device)

    def from_image(self, filename: str = None):
        self.logger.log_info(f'Source image: {self.source}')
        self.logger.log_info(f'Target image: {self.target}')

        target = cv2.imread(self.target, cv2.IMREAD_COLOR)
        source = cv2.imread(self.source, cv2.IMREAD_COLOR)
        
        if self.options.channels == 1:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        self.logger.log_info('Cropping, resizing, and extracting landmarks from source and target images...')
        target, target_landmark, _ = self.detect_crop_face(target, channels=self.options.channels, landmark_type=self.options.landmark_type, padding=self.options.padding, output_res=(self.options.image_size, self.options.image_size), face_alignment=self.fa)
        source, source_landmark, _ = self.detect_crop_face(source, channels=self.options.channels, landmark_type=self.options.landmark_type, padding=self.options.padding, output_res=(self.options.image_size, self.options.image_size), face_alignment=self.fa)

        if (target is None and target_landmark is None) or (source is None and source_landmark is None):
            self.logger.log_info('Could not find any faces!')
            return

        if self.options.channels == 1:
            target, target_landmark = target[:,:,None], target_landmark[:,:,None]
            source, source_landmark = source[:,:,None], source_landmark[:,:,None]

        target = np.ascontiguousarray(target.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
        target = torch.from_numpy(target * (1.0 / 255.0)).to(self.options.device)
        target = normalize(target, mean=self.options.normalize[0], std=self.options.normalize[1])

        source = np.ascontiguousarray(source.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
        source = torch.from_numpy(source * (1.0 / 255.0)).to(self.options.device)
        source = normalize(source, mean=self.options.normalize[0], std=self.options.normalize[1])

        source_landmark = np.ascontiguousarray(source_landmark.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
        source_landmark = torch.from_numpy(source_landmark * (1.0 / 255.0)).to(self.options.device)
        source_landmark = normalize(source_landmark, mean=self.options.normalize[0], std=self.options.normalize[1])

        self.logger.log_info('Applying facial reenactment...')
        output =  self.network(target, source_landmark)
        image = torch.cat((target, source, source_landmark, output), dim=0)
        if filename is None:
            filename = f't_{datetime.now():%Y%m%d_%H%M%S}'
        else:
            filename = f'{filename}_t_{datetime.now():%Y%m%d_%H%M%S}'
        self.logger.save_image(self.options.output_dir, filename, image, nrow=self.options.batch_size)
        self.logger.log_info(f'Facial reenactment done. Image saved in {os.path.join(self.options.output_dir, filename)}.')


    def from_video(self, filename: str = None, output_path: str = None):
        self.logger.log_info(f'Source image: {self.source}')
        self.logger.log_info(f'Target video: {self.target}')

        # SOURCE IMAGE PREPROCESSING
        self.logger.log_info('Cropping, resizing, and extracting landmarks from source image...')
        target = cv2.imread(self.source, cv2.IMREAD_COLOR)

        if self.options.channels == 1:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        target, target_landmark, _ = self.detect_crop_face(target, channels=self.options.channels, landmark_type=self.options.landmark_type, padding=self.options.padding, output_res=(self.options.image_size, self.options.image_size), face_alignment=self.fa)
        if self.options.channels == 1:
            target, target_landmark = target[:,:,None], target_landmark[:,:,None]

        if target is None and target_landmark is None:
            self.logger.log_info('Could not find any faces in source image!')
            return

        target = np.ascontiguousarray(target.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
        target = torch.from_numpy(target * (1.0 / 255.0)).to(self.options.device)
        target = normalize(target, mean=self.options.normalize[0], std=self.options.normalize[1])

        # TARGET VIDEO PREPROCESSING
        self.logger.log_info('Cropping, resizing, and extracting landmarks from target video...')
        source_frames = np.concatenate([extract_frames(self.target)])
        source_frames_filtered = []
        source_landmarks = []
        for i, source_frame in enumerate(source_frames):
            if self.options.channels == 1:
                source_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
            f, l, _ = self.detect_crop_face(source_frame, channels=self.options.channels, landmark_type=self.options.landmark_type, padding=self.options.padding, output_res=(self.options.image_size, self.options.image_size), face_alignment=self.fa)
            if self.options.channels == 1:
                f, l = f[:,:,None], l[:,:,None]

            if f is not None and l is not None:
                if self.options.channels == 3:
                    f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                f = f.transpose(2, 0 ,1) * (1.0 / 255.0)
                f = normalize(f, mean=self.options.normalize[0], std=self.options.normalize[1])
                source_frames_filtered.append(f)
                source_landmarks.append(l)
                print(f'[{i + 1}/{len(source_frames)}] done', end='\r')

        if len(source_landmarks) == 0:
            self.logger.log_info('Could not find any faces in target video!')
            return

        self.logger.log_info('Applying facial reenactment...')
        outputs = []
        for i, source_landmark in enumerate(source_landmarks):
            source_landmark = np.ascontiguousarray(source_landmark.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))
            source_landmark = torch.from_numpy(source_landmark * (1.0 / 255.0)).to(self.options.device)
            source_landmark = normalize(source_landmark, mean=self.options.normalize[0], std=self.options.normalize[1])

            source_landmarks[i] = source_landmark.detach().cpu().numpy()[0, :, :, :]

            output = self.network(target, source_landmark).detach().cpu().numpy()[0, :, :, :]
            outputs.append(output)
            print(f'[{i + 1}/{len(source_landmarks)}] done', end='\r')

        # CONCAT IMAGES
        self.logger.log_info('Processing video...')
        outputs_cat = []
        target = target.detach().cpu().numpy()[0, :, :, :]
        for i, (s, l, o) in enumerate(zip(source_frames_filtered, source_landmarks, outputs)):
            frame = torch.from_numpy(np.concatenate((target, s, l, o), axis=2)).to(self.options.device)
            o = self.logger.save_image(path=None, filename=None, image=frame, ret_image=True)
            outputs_cat.append(o)

        # SAVE VIDEO
        self.logger.log_info('Saving video...')
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

        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=25.0, frameSize=outputs_cat[0].shape[:2][::-1])
        for i, o in enumerate(outputs_cat):
            video_writer.write(o)
            print(f'[{i + 1}/{len(outputs)}] done', end='\r')
        video_writer.release()

        self.logger.log_info(f'Facial reenactment done. Video saved in {filename}.')


    def detect_crop_face(self, frame, channels, landmark_type, padding, output_res, face_alignment):
        frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        landmarks = face_alignment.get_landmarks_from_image(frame)
        face_detected = landmarks is not None
        if not face_detected:
            return None, None, None
        else:
            landmarks = landmarks[0]
            frame = crop_frame(frame, landmarks, (224,224), padding, method='cv2')
            if frame is None:
                return None, None, None
            else:
                landmarks = face_alignment.get_landmarks_from_image(frame)
                face_detected = landmarks is not None
                if not face_detected:
                    return None, None, None
                else:
                    landmarks_plt = plot_landmarks(landmarks[0], landmark_type=landmark_type, channels=channels, output_res=(frame.shape[0], frame.shape[1]), input_res=frame.shape[:2])
                    landmarks_plt = cv2.resize(landmarks_plt, output_res)
                    frame = cv2.resize(frame, output_res)
                    return frame, landmarks_plt, landmarks[0]
