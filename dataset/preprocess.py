import os
import shutil
import PIL
import cv2
import numpy as np
import pandas as pd

from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from configs import DatasetOptions
from loggings.logger import Logger
from dataset.utils import crop_frame, extract_frames, plot_landmarks

##### START PreprocessVoxCeleb #####
class PreprocessVoxCeleb():
    def __init__(self, logger: Logger, options: DatasetOptions):
        self.logger = logger
        self.options = options

        self.num_pairs = self.options.num_pairs
        self.max_frames = self.options.max_frames

        self.padding = self.options.padding
        self.padding_color = self.options.padding_color
        self.output_res = (self.options.image_size_db,self.options.image_size_db)
        self.prune_videos = self.options.prune_videos

        if not os.path.isdir(self.options.output):
            os.makedirs(self.options.output)

        self.logger.log_info('===== DATASET PRE-PROCESSING VOXCELEB2 =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Saving K random frames from each video (K = {self.num_pairs}).')


    def preprocess(self):
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)
        init_pool(fa, self.options.output)

        self._create_csv(self.options.csv, self.num_pairs)

        # Prune videos to large to fit into RAM
        if self.prune_videos:
            prune_videos(self.options.source, max_num_videos=100, max_bytes_videos=30_000_000)

        # Get video paths to be processed
        self.video_list = self._get_video_list(
            self.options.source,
            self.options.num_videos,
            self.options.output,
            overwrite=self.options.overwrite_videos
        )
        self.logger.log_info(f'Processing {len(self.video_list)} videos...')

        # Start processing
        self.counter = 1
        for v in self.video_list:
            start_time = datetime.now()
            self.process_video_folder(v, self.options.csv, num_frames=self.num_pairs)
            self.logger.log_info(f'{self.counter}/{len(self.video_list)}: {v[0]} saved | Time: {datetime.now() - start_time}')
            self.counter += 1
        self.logger.log_info(f'All {len(self.video_list)} videos processed.')
        self.logger.log_info(f'CSV {self.options.csv} created.')

        sanitize_csv(self.options.csv, 'id')


    def preprocess_by_ids(self, ids):
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)
        init_pool(fa, self.options.output)

        self._create_csv(self.options.csv, self.num_pairs)

        # Prune videos to large to fit into RAM
        if self.prune_videos:
            prune_videos(self.options.source, max_num_videos=100, max_bytes_videos=30_000_000)

        # Get video paths to be processed
        self.video_list = self._get_video_list_id(
            self.options.source,
            self.options.output,
            overwrite=self.options.overwrite_videos,
            ids=ids
        )
        self.logger.log_info(f'Processing {len(self.video_list)} videos...')

        # Start processing
        self.counter = 1
        for v in self.video_list:
            start_time = datetime.now()
            self.process_video_folder_id(v, self.options.csv, num_frames=self.num_pairs, max_frames=self.max_frames)
            self.logger.log_info(f'{self.counter}/{len(self.video_list)}: {v[0]} saved | Time: {datetime.now() - start_time}')
            self.counter += 1
        self.logger.log_info(f'All {len(self.video_list)} videos processed.')
        self.logger.log_info(f'CSV {self.options.csv} created.')

        # Extend dataset
        video_limit = 2
        _ids = os.listdir(os.path.join(self.options.source, 'mp4'))
        misc_ids = [i for i in _ids if i not in ids]

        self.video_list = self._get_video_list_id(
            self.options.source,
            self.options.output,
            overwrite=self.options.overwrite_videos,
            ids=misc_ids,
            video_limit=video_limit
        )
        self.logger.log_info(f'Extending dataset with additional {len(self.video_list)} data...')

        self.counter = 1
        for v in self.video_list:
            start_time = datetime.now()
            self.process_video_folder_id(v, self.options.csv, num_frames=self.num_pairs, max_frames=self.max_frames, video_limit=video_limit)
            self.logger.log_info(f'{self.counter}/{len(self.video_list)}: {v[0]} saved | Time: {datetime.now() - start_time}')
            self.counter += 1
        
        self.logger.log_info(f'All {len(self.video_list)} videos processed.')
        self.logger.log_info(f'CSV {self.options.csv} updated.')

        sanitize_csv(self.options.csv, 'id')


    def _create_csv(self, csv_path, num_frames):
        if not os.path.isfile(csv_path):
            self.logger.log_info(f'Creating CSV file {csv_path}).')

            header = []
            for i in range(num_frames):
                header.append(f'landmark{i+1}')
                header.append(f'image{i+1}')
            
            header.append('id')
            header.append('id_video\n')

            with open(csv_path, 'a') as csv_file:
                csv_file.write(','.join(header))
        else:
            self.logger.log_info(f'CSV file {csv_path} loaded.')


    def _get_video_list(self, source, size, output, overwrite=True):
        self.logger.log_info(f'Analyze already processed videos...')

        already_processed = []
        if not overwrite:
            for root, dirs, files in os.walk(output):
                if len(files) > 0 and len(dirs) <= 0:
                    already_processed.append(os.path.sep.join(root.split(os.path.sep)[-2:]))
        
        self.logger.log_info(f'{len(already_processed)} videos already processed.')
        self.logger.log_info(f'Analyze videos to be processed...')
        video_list = []

        # Process n=size videos
        counter = 0
        for root, dirs, files in os.walk(source):
            if len(files) > 0 and os.path.sep.join(root.split(os.path.sep)[-2:]) not in already_processed:
                assert contains_only_videos(files) and len(dirs) == 0
                video_list.append((root, files))
                counter += 1
                if 0 < size <= counter:
                    break

        return video_list

    
    def _get_video_list_id(self, source, output, overwrite=True, ids=None, video_limit=None):
        self.logger.log_info(f'Analyze already processed videos...')

        already_processed = []
        if not overwrite:
            for root, dirs, files in os.walk(output):
                if len(files) > 0 and len(dirs) <= 0:
                    already_processed.append(os.path.sep.join(root.split(os.path.sep)[-2:]))
        
        self.logger.log_info(f'{len(already_processed)} videos already processed.')

        if video_limit is not None:
            processed_dict = {f'{p.split(os.path.sep)[-2]}': 0 for p in already_processed}
            for p in already_processed:
                id_ = p.split(os.path.sep)[-2]
                if id_ in processed_dict:
                    processed_dict[id_] += 1

        self.logger.log_info(f'Analyze videos to be processed...')
        video_list = []

        # Process videos by id
        for id_ in ids:
            video_counter = 0
            id_path = os.path.join(source, 'mp4', id_)
            if video_limit is not None and id_ in processed_dict and processed_dict[id_] >= 2:
                continue
            for root, dirs, files in os.walk(id_path):
                if len(files) > 0 and os.path.sep.join(root.split(os.path.sep)[-2:]) not in already_processed:
                    assert contains_only_videos(files) and len(dirs) == 0
                    video_list.append((root, files))
                    video_counter += 1
                    if video_limit is not None and video_counter >= video_limit:
                        break

        return video_list


    def process_video_folder(self, video, csv, num_frames):
        folder, files = video
        identity = folder.split(os.path.sep)[-2:][0]
        video_id = folder.split(os.path.sep)[-2:][1]

        try:
            assert contains_only_videos(files)
            frames_total = np.concatenate([extract_frames(os.path.join(folder, f)) for f in files])
            self.save_video(
                frames=select_random_frames(frames_total, num_frames),
                identity=identity,
                video_id=video_id,
                path=_OUT_DIR,
                csv=csv,
                face_alignment=_FA
            )
        except Exception as e:
            self.logger.log_error(f'Video {identity}/{video_id} could not be processed:\n{e}')


    def process_video_folder_id(self, video, csv, num_frames, max_frames, video_limit=None):
        folder, files = video
        identity = folder.split(os.path.sep)[-2:][0]
        video_id = folder.split(os.path.sep)[-2:][1]

        try:
            assert contains_only_videos(files)
            frames_total = np.concatenate([extract_frames(os.path.join(folder, f)) for f in files])
            
            np.random.shuffle(frames_total)
            end_idx = frames_total.shape[0] - (frames_total.shape[0] % num_frames)
            end_idx = max_frames if end_idx > max_frames else end_idx
            selected_frames = frames_total[0:end_idx]
            # Split array in chunks of size num_frames=3
            triplets = np.array_split(selected_frames, selected_frames.shape[0] // num_frames)
            for idx, triplet in enumerate(triplets):
                self.save_video(
                    frames=triplet,
                    identity=identity,
                    video_id=video_id,
                    path=_OUT_DIR,
                    csv=csv,
                    face_alignment=_FA,
                    frame_id=idx,
                    frames_total=frames_total,
                    frame_limit=1 if video_limit is not None else None
                )
                if video_limit is not None:
                    break
        except Exception as e:
            self.logger.log_error(f'Video {identity}/{video_id} could not be processed:\n{e}')

    
    def save_video(self, path, csv, identity, video_id, frames, face_alignment, frame_id=None, frames_total=None, frame_limit=None):
        path = os.path.join(path, identity, video_id)
        if not os.path.isdir(path):
            os.makedirs(path)
        
        csv_line = []

        if frame_id is None:
            r = range(len(frames))
        elif frame_id is not None:
            r = range(frames.shape[0])
        elif frame_limit is not None:
            r = range(frame_limit)

        for i in r:
            x = frames[i]
            x, y = detect_face(x, frames_total, face_alignment, self.padding, self.padding_color, self.output_res)

            filename_y = f'{i+1}.npy' if frame_id is None else f'{frame_id+1}_{i+1}.npy'
            csv_line.append(filename_y)
            np.save(os.path.join(path, filename_y), y)

            x = PIL.Image.fromarray(x, 'RGB')
            filename_x = f'{i+1}.png' if frame_id is None else f'{frame_id+1}_{i+1}.png'
            csv_line.append(filename_x)
            x.save(os.path.join(path, filename_x))

        csv_line.append(identity)
        csv_line.append(video_id + '\n')

        with open(csv, 'a') as csv_file:
            csv_file.write(','.join(csv_line))
##### END PreprocessVoxCeleb #####


##### START PreprocessFaceForensics #####
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


    def preprocess(self):
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)
        init_pool(fa, self.options.output)

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

##### END PreprocessFaceForensics #####

def init_pool(face_alignment, output):
    global _FA
    _FA = face_alignment
    global _OUT_DIR
    _OUT_DIR = output


def sanitize_csv(csv_path: str, sort_col: str):
    df = pd.read_csv(csv_path, sep=',', header=0)
    df = df.drop_duplicates(keep=False)
    df = df.sort_values(by=[sort_col], ascending=True)
    df.to_csv(csv_path, sep=',', index=False)


def prune_videos(source, max_num_videos=None, max_bytes_videos=None):
    # Split large videos into chunks (files length)
    if max_num_videos is not None:
        for root, dirs, files in os.walk(source):
            if len(files) >= max_num_videos and len(dirs) <= 0:
                split_large_video(root, files, chunk_size=50)

    # Split large videos into chunks (files size)
    if max_bytes_videos is not None:
        for root, dirs, files in os.walk(source):
            if len(files) > 0 and len(dirs) <= 0:
                total_size = 0
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
                if total_size >= max_bytes_videos:
                    split_large_video(root, files, chunk_size=len(files)//3)


def split_large_video(source, files, chunk_size):
    files = sorted(files)
    chunks = divide_chunks(files, chunk_size)

    for i, chunk in enumerate(chunks):
        destination = source + f'_{i+1}'

        if not os.path.isdir(destination):
            os.makedirs(destination)
        
        for f in chunk:
            shutil.move(os.path.join(source, f), destination)
    #shutil.rmtree(source)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i+n]


def contains_only_videos(files, extension='.mp4'):
        return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


def select_random_frames(frames, num_frames):
    idx = np.random.choice(len(frames)-1, num_frames, replace=False)
    return [frames[sample] for sample in idx]


def crop_face(frame, frames_total, fa, padding, padding_color, output_res, method='pyplot'):
    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_color)
    landmarks = fa.get_landmarks_from_image(frame)
    face_detected = landmarks is not None
    if not face_detected:
        id_x = np.random.randint(len(frames_total))
        frame = frames_total[id_x]
        return crop_face(frame, frames_total, fa, padding, padding_color, output_res, method)
    else:
        landmarks = landmarks[0]
        frame = crop_frame(frame, landmarks, output_res, padding, method=method)
        if frame is None:
            id_x = np.random.randint(len(frames_total))
            frame = frames_total[id_x]
            return crop_face(frame, frames_total, fa, padding, padding_color, output_res, method)
        else:
            return frame


def detect_face(frame, frames_total, fa, padding, padding_color, output_res, method='pyplot'):
    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_color)
    landmarks = fa.get_landmarks_from_image(frame)
    face_detected = landmarks is not None
    if not face_detected:
        id_x = np.random.randint(len(frames_total))
        frame = frames_total[id_x]
        return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
    else:
        landmarks = landmarks[0]
        frame = crop_frame(frame, landmarks, output_res, padding, method=method)
        if frame is None:
            id_x = np.random.randint(len(frames_total))
            frame = frames_total[id_x]
            return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
        else:
            landmarks = fa.get_landmarks_from_image(frame)
            face_detected = landmarks is not None
            if not face_detected:
                id_x = np.random.randint(len(frames_total))
                frame = frames_total[id_x]
                return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
            else:
                return frame, landmarks[0]
