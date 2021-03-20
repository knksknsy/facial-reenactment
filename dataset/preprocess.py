import os
import shutil
import PIL
import cv2
import numpy as np
import pandas as pd

from multiprocessing import Pool
from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from configs import DatasetOptions
from loggings.logger import Logger
from dataset.dataset import plot_landmarks

class Preprocess():
    def __init__(self, logger: Logger, options: DatasetOptions):
        self.logger = logger
        self.options = options
        self.ids = self.options.vox_ids if hasattr(self.options, 'vox_ids') else None
        self.num_pairs = self.options.num_pairs
        self.max_frames = self.options.max_frames
        self.padding = self.options.padding
        self.padding_color = self.options.padding_color
        self.output_res = (self.options.image_size_db,self.options.image_size_db)
        self.prune_videos = self.options.prune_videos
        if not os.path.isdir(self.options.output):
            os.makedirs(self.options.output)


    def preprocess_dataset(self):

        self.logger.log_info('===== DATASET PRE-PROCESSING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Saving K random frames from each video (K = {self.num_pairs}).')
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)

        # pool = Pool(processes=4, initializer=self._init_pool, initargs=(fa, output))
        # pool.map(self._process_video_folder, self.video_list)
        self._init_pool(fa, self.options.output)

        self._create_csv(self.options.csv, self.num_pairs)

        # Prune videos to large to fit into RAM
        if self.prune_videos:
            self._prune_videos(self.options.source)

        self.video_list = self._get_video_list(
            self.options.source,
            self.options.num_videos,
            self.options.output,
            overwrite=self.options.overwrite_videos,
            ids=self.ids
        )
        self.logger.log_info(f'Processing {len(self.video_list)} videos...')

        self.counter = 1
        for v in self.video_list:
            start_time = datetime.now()
            self._process_video_folder(v, self.options.csv, self.num_pairs, ids=self.ids)
            self.logger.log_info(f'{self.counter}/{len(self.video_list)}: {v[0]} saved | Time: {datetime.now() - start_time}')
            self.counter += 1

        self.logger.log_info(f'All {len(self.video_list)} videos processed.')
        self.logger.log_info(f'CSV {self.options.csv} created.')

        # Extend dataset
        if self.ids is not None:
            video_limit = 2
            ids = os.listdir(os.path.join(self.options.source, 'mp4'))
            misc_ids = [i for i in ids if i not in self.ids]

            self.video_list = self._get_video_list(
                self.options.source,
                self.options.num_videos,
                self.options.output,
                overwrite=self.options.overwrite_videos,
                ids=misc_ids,
                video_limit=video_limit
            )
            self.logger.log_info(f'Extending dataset with additional {len(self.video_list)} data...')

            self.counter = 1
            for v in self.video_list:
                start_time = datetime.now()
                self._process_video_folder(v, self.options.csv, self.num_pairs, ids=misc_ids, video_limit=video_limit)
                self.logger.log_info(f'{self.counter}/{len(self.video_list)}: {v[0]} saved | Time: {datetime.now() - start_time}')
                self.counter += 1
            
            self.logger.log_info(f'All {len(self.video_list)} videos processed.')
            self.logger.log_info(f'CSV {self.options.csv} updated.')

        self._sanitize_csv()


    def _create_csv(self, path, num_frames):
        if not os.path.isfile(path):
            self.logger.log_info(f'Creating CSV file {path}).')

            header = []
            for i in range(num_frames):
                header.append(f'landmark{i+1}')
                header.append(f'image{i+1}')
            
            header.append('id')
            header.append('id_video\n')

            with open(path, 'a') as csv_file:
                csv_file.write(','.join(header))
        else:
            self.logger.log_info(f'CSV file {path} loaded.')


    def _sanitize_csv(self):
        df = pd.read_csv(self.options.csv, sep=',', header=0)
        df = df.drop_duplicates(keep=False)
        df = df.sort_values(by=['id'], ascending=True)
        df.to_csv(self.options.csv, sep=',', index=False)


    def _prune_videos(self, source):
        self.logger.log_info('Pruning videos...')
        pruned = False

        # Split large videos into chunks (files length)
        max_videos_n = 100
        for root, dirs, files in os.walk(source):
            if len(files) >= max_videos_n and len(dirs) <= 0:
                pruned = True
                self._split_large_video(root, files, 50)

        # Split large videos into chunks (files size)
        max_video_bytes = 30000000
        for root, dirs, files in os.walk(source):
            if len(files) > 0 and len(dirs) <= 0:
                pruned = True
                total_size = 0
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
                if total_size >= max_video_bytes:
                    self._split_large_video(root, files, len(files)//3)

        if pruned:
            self.logger.log_info('Videos pruned.')
        else:
            self.logger.log_info('No videos to be pruned.')


    def _split_large_video(self, source, files, chunk_size):
        files = sorted(files)
        chunks = self._divide_chunks(files, chunk_size)

        for i, chunk in enumerate(chunks):
            destination = source + f'_{i+1}'

            if not os.path.isdir(destination):
                os.makedirs(destination)
            
            for f in chunk:
                shutil.move(os.path.join(source, f), destination)
        #shutil.rmtree(source)


    def _divide_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i+n]


    def _contains_only_videos(self, files, extension='.mp4'):
        return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


    def _get_video_list(self, source, size, output, overwrite=True, ids=None, video_limit=None):
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
        if ids is not None:
            for id_ in ids:
                video_counter = 0
                id_path = os.path.join(source, 'mp4', id_)
                if video_limit is not None and id_ in processed_dict and processed_dict[id_] >= 2:
                    continue
                for root, dirs, files in os.walk(id_path):
                    if len(files) > 0 and os.path.sep.join(root.split(os.path.sep)[-2:]) not in already_processed:
                        assert self._contains_only_videos(files) and len(dirs) == 0
                        video_list.append((root, files))
                        video_counter += 1
                        if video_limit is not None and video_counter >= video_limit:
                            break

        # Process n=size videos
        else:
            counter = 0
            for root, dirs, files in os.walk(source):
                if len(files) > 0 and os.path.sep.join(root.split(os.path.sep)[-2:]) not in already_processed:
                    assert self._contains_only_videos(files) and len(dirs) == 0
                    video_list.append((root, files))
                    counter += 1
                    if 0 < size <= counter:
                        break

        return video_list


    def _init_pool(self, face_alignment, output):
        global _FA
        _FA = face_alignment
        global _OUT_DIR
        _OUT_DIR = output


    def _process_video_folder(self, video, csv, num_frames, ids=None, video_limit=None):
        folder, files = video
        identity = folder.split(os.path.sep)[-2:][0]
        video_id = folder.split(os.path.sep)[-2:][1]

        try:
            assert self._contains_only_videos(files)
            frames_total = np.concatenate([self._extract_frames(os.path.join(folder, f)) for f in files])

            if ids is not None:
                np.random.shuffle(frames_total)
                end_idx = frames_total.shape[0] - (frames_total.shape[0] % num_frames)
                end_idx = self.max_frames if end_idx > self.max_frames else end_idx
                selected_frames = frames_total[0:end_idx]
                # Split array in chunks of size num_frames=3
                triplets = np.array_split(selected_frames, selected_frames.shape[0] // num_frames)
                for idx, triplet in enumerate(triplets):
                    self._save_video(
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
            else:
                self._save_video(
                    frames=self._select_random_frames(frames_total, num_frames),
                    identity=identity,
                    video_id=video_id,
                    path=_OUT_DIR,
                    csv=csv,
                    face_alignment=_FA
                )

        except Exception as e:
            self.logger.log_error(f'Video {identity}/{video_id} could not be processed:\n{e}')


    def _extract_frames(self, video):
        cap = cv2.VideoCapture(video)

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

        fn, ret = 0, True
        while fn < n_frames and ret:
            ret, img = cap.read()
            frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fn += 1

        cap.release()
        return frames


    def _select_random_frames(self, frames, num_frames):
        S = []
        while len(S) < num_frames:
            s = np.random.randint(0, len(frames)-1)
            if s not in S:
                S.append(s)

        return [frames[s] for s in S]


    def _save_video(self, path, csv, identity, video_id, frames, face_alignment, frame_id=None, frames_total=None, frame_limit=None):
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
            x, y = self.detect_face(x, frames_total, face_alignment)

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


    def crop_frame(self, frame, landmarks, dimension, padding):
        heatmap = plot_landmarks(landmarks=landmarks, landmark_type='boundary', channels=3, output_res=frame.shape[0], input_res=frame.shape[0])

        rows = np.any(heatmap, axis=1)
        cols = np.any(heatmap, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        frame = frame[rmin-padding:rmax+padding, cmin-padding:cmax+padding]
        frame = cv2.resize(frame, dimension)
        return frame


    def detect_face(self, frame, frames_total, fa):
        frame = cv2.copyMakeBorder(frame, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, value=self.padding_color)
        landmarks = fa.get_landmarks_from_image(frame)
        face_detected = landmarks is not None
        if not face_detected:
            id_x = np.random.randint(len(frames_total))
            frame = frames_total[id_x]
            self.detect_face(frame, frames_total, fa)
        else:
            landmarks = landmarks[0]
            frame = self.crop_frame(frame, landmarks, self.output_res, self.padding)
            landmarks = fa.get_landmarks_from_image(frame)
            face_detected = landmarks is not None
            if not face_detected:
                id_x = np.random.randint(len(frames_total))
                frame = frames_total[id_x]
                self.detect_face(frame, frames_total, fa)
            else:
                return frame, landmarks[0]
