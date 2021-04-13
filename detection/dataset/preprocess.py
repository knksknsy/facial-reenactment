import os
import PIL
import numpy as np

from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from configs.dataset_options import DatasetOptions
from loggings.logger import Logger
from utils.preprocess import prune_videos, sanitize_csv, extract_frames, select_random_frames, detect_face, crop_face


class PreprocessFaceForensics():
    def __init__(self, logger: Logger, options: DatasetOptions, method_list):
        self.logger = logger
        self.options = options
        self.method_list = method_list

        self.max_frames = self.options.max_frames

        self.padding = self.options.padding
        self.padding_color = self.options.padding_color
        self.output_res = (self.options.image_size_db,self.options.image_size_db)
        self.prune_videos = self.options.prune_videos

        if not os.path.isdir(self.options.output):
            os.makedirs(self.options.output)

        self.logger.log_info('===== DATASET PRE-PROCESSING FACEFORENSICS =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Saving K random frames from each video (K = {self.max_frames}).')


    def init_pool(self, face_alignment, output):
        global _FA
        _FA = face_alignment
        global _OUT_DIR
        _OUT_DIR = output


    def preprocess(self):
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)
        self.init_pool(fa, self.options.output)

        self._create_csv(self.options.csv)

        # Prune videos to large to fit into RAM
        if self.prune_videos:
            prune_videos(self.options.source, max_bytes_videos=30_000_000)
        
        # Get video paths to be processed
        self.video_list = self._get_video_list(
            self.options.source,
            self.options.num_videos,
            self.options.output,
            self.method_list,
            overwrite=self.options.overwrite_videos
        )
        self.logger.log_info(f'Preprocessing {len(self.video_list)} videos...')

        # Start preprocessing
        self.counter = 1
        for v in self.video_list:
            start_time = datetime.now()
            self.process_video_folder(v, self.options.csv, self.options.output, num_frames=self.max_frames)
            self.logger.log_info(f"{self.counter}/{len(self.video_list)}: {v['output_fake_path']} saved | Time: {datetime.now() - start_time}")
            self.counter += 1
        self.logger.log_info(f'All {len(self.video_list)} videos processed.')
        self.logger.log_info(f'CVS {self.options.csv} created.')

        sanitize_csv(self.options.csv, sort_col='image_real')


    def _create_csv(self, csv_path):
        if not os.path.isfile(csv_path):
            self.logger.log_info(f'Creating CSV file {csv_path}.')

            header = ['image_real', 'label_real', 'id_real', 'image_fake', 'label_fake', 'id_fake', 'landmark_fake', 'method\n']
            with open(csv_path, 'a') as csv_file:
                csv_file.write(','.join(header))
        else:
            self.logger.log_info(f'CSV file {csv_path} loaded.')


    def _get_video_list(self, source, size, output, method_list, overwrite=True):
        self.logger.log_info('Analyze already processed videos...')
        already_processed = []
        if not overwrite:
            for root, dirs, files in os.walk(output):
                if len(files) > 0 and len(dirs) <= 0:
                    processed_path = root.replace(output,'')
                    processed_path = processed_path[1:] if processed_path[0] == os.path.sep else processed_path
                    already_processed.append(processed_path)
        self.logger.log_info(f'{len(already_processed)} videos already processed.')

        self.logger.log_info('Analyze videos to be processed...')
        video_list = []

        # Process n=size videos
        counter = 0
        for root, dirs, files in os.walk(source):
            if len(files) > 0 and len(dirs) <= 0 and 'original_sequences' not in root:
                for file in files:
                    method = root.split(os.path.sep)[-3]
                    if not file.startswith('.') and method in method_list:
                        id_real = 'id' + file.replace('.mp4','').split('_')[0]
                        id_fake = 'id' + file.replace('.mp4','').split('_')[1]
                        filename_real = id_real.replace('id','') + '.mp4'
                        filename_fake = file
                        source_fake_path = root
                        source_real_path = root.replace('manipulated_sequences', 'original_sequences').replace(method, 'youtube')
                        output_real_path = os.path.join(method, 'original_sequences', id_real)
                        output_fake_path = os.path.join(method, 'manipulated_sequences', id_real)
                        params = {
                            'id_real': id_real,
                            'id_fake': id_fake,
                            'filename_real': filename_real,
                            'filename_fake': filename_fake,
                            'source_fake_path': source_fake_path,
                            'source_real_path': source_real_path,
                            'output_real_path': output_real_path,
                            'output_fake_path': output_fake_path,
                            'method': method
                        }
                        if output_real_path not in already_processed or output_fake_path not in already_processed:
                            video_list.append(params)
                        counter += 1
                        if 0 < size <= counter:
                            break
        return video_list


    def process_video_folder(self, video, csv, output_path, num_frames):
        id_real, id_fake, method = video['id_real'], video['id_fake'], video['method']
        filename_real, filename_fake = video['filename_real'], video['filename_fake']
        source_real_path, source_fake_path = video['source_real_path'], video['source_fake_path']
        output_real_path, output_fake_path = video['output_real_path'], video['output_fake_path']

        frames_fake_total = np.concatenate([extract_frames(os.path.join(source_fake_path, filename_fake))])
        frames_real_total = np.concatenate([extract_frames(os.path.join(source_real_path, filename_real))])
        num_frames = num_frames if frames_fake_total.shape[0] > num_frames else frames_fake_total.shape[0]

        image_fake, image_real = None, None
        frames_fake = select_random_frames(frames_fake_total, num_frames)
        frames_real = select_random_frames(frames_real_total, num_frames)

        # Extract frames from videos
        for i, (frame_fake, frame_real) in enumerate(zip(frames_fake, frames_real)):
            cur_filename_fake = filename_fake.replace('.mp4', f'_{i}.png')
            cur_filename_real = filename_real.replace('.mp4', f'_{i}.png')
            cur_filename_landmark = filename_fake.replace('.mp4', f'_{i}.npy')

            image_fake, landmark_fake = detect_face(frame_fake, frames_fake_total, fa=_FA, padding=self.padding, padding_color=self.padding_color, output_res=self.output_res, method='cv2')
            self.save_frame(image_fake, os.path.join(output_path, output_fake_path), cur_filename_fake)
            np.save(os.path.join(output_path, output_fake_path, cur_filename_landmark), landmark_fake)

            image_real = crop_face(frame_real, frames_real_total, fa=_FA, padding=self.padding, padding_color=self.padding_color, output_res=self.output_res, method='cv2')
            self.save_frame(image_real, os.path.join(output_path, output_real_path), cur_filename_real)

            csv_line = [os.path.join(output_real_path, cur_filename_real), '1', id_real, os.path.join(output_fake_path, cur_filename_fake), '0', id_fake, os.path.join(output_fake_path, cur_filename_landmark), method + '\n']
            with open(csv, 'a') as csv_file:
                csv_file.write(','.join(csv_line))

            self.logger.log_info(f'\t[{i+1}/{num_frames}] saved into: {os.path.join(output_fake_path, cur_filename_fake)}')


    def save_frame(self, frame, dest_path, filename):
        if not os.path.isdir(dest_path):
            os.makedirs(dest_path)

        frame = PIL.Image.fromarray(frame, 'RGB')
        frame.save(os.path.join(dest_path, filename))
