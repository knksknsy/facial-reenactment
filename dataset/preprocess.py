import os
import sys
import logging
import shutil
import random
import PIL
import cv2
import numpy as np

from multiprocessing import Pool
from datetime import datetime
from face_alignment import FaceAlignment, LandmarksType

from configs import DatasetOptions

class Preprocess():
    def __init__(self, options: DatasetOptions):
        self.options = options

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self._preprocess_dataset()
    
    def _preprocess_dataset(self):
        logging.info('===== DATASET PRE-PROCESSING =====')
        logging.info(f'Running on {self.options.device.upper()}.')
        logging.info(f'Saving K+1 random frames from each video (K = {self.options.frames + 1}).')
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)
        
        self._create_csv(self.options.csv)
        # Prune videos to large to fit into RAM
        self._prune_videos(self.options.source)

        video_list = self._get_video_list(
            self.options.source,
            self.options.size,
            self.options.output,
            overwrite=self.options.overwrite
        )

        logging.info(f'Processing {len(video_list)} videos...')
        # pool = Pool(processes=4, initializer=self._init_pool, initargs=(fa, output))
        # pool.map(self._process_video_folder, video_list)

        self._init_pool(fa, self.options.output)
        counter = 1
        for v in video_list:
            start_time = datetime.now()
            self._process_video_folder(v, self.options.csv)
            logging.info(f'{counter}/{len(video_list)}\t{datetime.now()-start_time}')
            counter += 1

        logging.info(f'All {len(video_list)} videos processed.')
        logging.info(f'CSV {self.options.csv} created.')

    
    def _create_csv(self, path):
        if not os.path.isfile(path):
            logging.info(f'Creating CSV file {path}).')

            header = []
            for i in range(self.options.frames + 1):
                header.append(f'landmark{i+1}')
                header.append(f'image{i+1}')
            
            header.append('id')
            header.append('id_video\n')

            with open(path, 'a') as csv_file:
                csv_file.write(','.join(header))
        else:
            logging.info(f'CSV file {path} loaded.')


    def _prune_videos(self, source):
        logging.info('Pruning videos...')
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
            logging.info('Videos pruned.')
        else:
            logging.info('No videos to be pruned.')


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
        """
        Checks whether the files provided all end with the specified video extension.
        :param files: List of file names.
        :param extension: Extension that all files should have.
        :return: True if all files end with the given extension.
        """
        return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


    def _get_video_list(self, source, size, output, overwrite=True):
        """
        Extracts a list of paths to videos to pre-process during the current run.

        :param source: Path to the root directory of the dataset.
        :param size: Number of videos to return.
        :param output: Path where the pre-processed videos will be stored.
        :param overwrite: If True, files that have already been processed will be overwritten, otherwise, they will be
        ignored and instead, different files will be loaded.
        :return: List of paths to videos.
        """
        logging.info(f'Analyze already processed videos...')

        already_processed = []
        if not overwrite:
            for root, dirs, files in os.walk(output):
                if len(files) > 0 and len(dirs) <= 0:
                    already_processed.append(os.path.sep.join(root.split(os.path.sep)[-2:]))
        
        logging.info(f'{len(already_processed)} videos already processed.')

        logging.info(f'Analyze videos to be processed... (May take a while)')
        video_list = []
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
    

    def _process_video_folder(self, video, csv):
        """
        Extracts all frames from a video, selects K+1 random frames, and saves them along with their landmarks.
        :param video: 2-Tuple containing (1) the path to the folder where the video segments are located and (2) the file
        names of the video segments.
        :param csv: Path where the CSV file will be stored.
        """
        folder, files = video

        try:
            assert self._contains_only_videos(files)
            frames = np.concatenate([self._extract_frames(os.path.join(folder, f)) for f in files])
            identity = folder.split(os.path.sep)[-2:][0]
            video_id = folder.split(os.path.sep)[-2:][1]

            self._save_video(
                frames=self._select_random_frames(frames),
                identity=identity,
                video_id=video_id,
                path=_OUT_DIR,
                csv=csv,
                face_alignment=_FA
            )
        except Exception as e:
            logging.error(f'Video {os.path.basename(os.path.normpath(folder))} could not be processed:\n{e}')
    

    def _extract_frames(self, video):
        """
        Extracts all frames of a video file. Frames are extracted in BGR format, but converted to RGB. The shape of the
        extracted frames is [height, width, channels]. Be aware that PyTorch models expect the channels to be the first
        dimension.
        :param video: Path to a video file.
        :return: NumPy array of frames in RGB.
        """
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
    

    def _select_random_frames(self, frames):
        """
        Selects K+1 random frames from a list of frames.
        :param frames: Iterator of frames.
        :return: List of selected frames.
        """
        S = []
        while len(S) <= self.options.frames:
            s = random.randint(0, len(frames)-1)
            if s not in S:
                S.append(s)

        return [frames[s] for s in S]
    

    def _save_video(self, path, csv, identity, video_id, frames, face_alignment):
        """
        Generates the landmarks for the face in each provided frame and saves the frames and the landmarks as a pickled
        list of dictionaries with entries {'frame', 'landmarks'}.

        :param path: Path to the output folder where the file will be saved.
        :param csv: Path where the CSV file will be stored.
        :param identity: ID of person.
        :param video_id: ID of video.
        :param frames: List of frames to save.
        :param face_alignment: Face Alignment model used to extract face landmarks.
        """
        path = os.path.join(path, identity, video_id)
        if not os.path.isdir(path):
            os.makedirs(path)
        
        csv_line = []

        for i in range(len(frames)):
            x = frames[i]
            y = face_alignment.get_landmarks_from_image(x)[0]
            filename_y = f'{i+1}.npy'
            csv_line.append(filename_y)
            np.save(os.path.join(path, filename_y), y)

            # # Save landmarks as image
            # plot = plot_landmarks(frames[i], y)
            # filename_plot = f'{i+1}_plot.png'
            # plot.save(os.path.join(path, filename_plot))

            x = PIL.Image.fromarray(x, 'RGB')
            filename_x = f'{i+1}.png'
            csv_line.append(filename_x)
            x.save(os.path.join(path, filename_x))

        csv_line.append(identity)
        csv_line.append(video_id + '\n')

        with open(csv, 'a') as csv_file:
            csv_file.write(','.join(csv_line))

        logging.info(f'Saved files: {path}')
