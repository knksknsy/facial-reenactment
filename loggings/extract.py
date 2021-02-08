import os
import glob
import struct
import cv2
import numpy as np
import pandas as pd
import tensorboard.compat.proto.event_pb2 as event_pb2

from datetime import datetime
from configs import LogsOptions
from loggings.logger import Logger

class LogsExtractor():
    def __init__(self, logger: Logger, options: LogsOptions):
        self.logger = logger
        self.options = options
        self()


    def __call__(self):
        self.run_start = datetime.now()
        self.logger.log_info('===== EXTRACTING LOGS =====')

        event_paths = glob.glob(os.path.join(self.options.log_dir, 'event*'))

        all_log = pd.DataFrame()
        for path in event_paths:
            log = self._sum_log(path)
            if log is not None:
                if all_log.shape[0] == 0:
                    all_log = log
                else:
                    all_log = all_log.append(log)

        self.logger.log_info(f'CSV shape: {all_log.shape}')
        all_log.head()
        filename = f'{datetime.now():%Y%m%d_%H%M%S}_metrics.csv'
        csv_path = os.path.join(self.options.output_dir, filename)
        all_log.to_csv(csv_path, index=None)
        self.logger.log_info(f'CSV saved to: {csv_path}')

        self.logger.log_info(f'Images saved to: {self.options.output_dir}')

        self.run_end = datetime.now()
        self.logger.log_info(f'Extracting logs finished in {self.run_end - self.run_start}.')


    def _sum_log(self, path):
        runlog = pd.DataFrame(columns=['metric', 'value', 'step'])
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(e)
            raise e

        while data:
            data, event_str = self._read_event(data)
            event = event_pb2.Event()

            event.ParseFromString(event_str)
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        r = {'metric': value.tag, 'value': value.simple_value, 'step': event.step}
                        runlog = runlog.append(r, ignore_index=True)
                    if value.HasField('image'):
                        img = value.image.encoded_image_string
                        img_array = np.asarray(bytearray(img), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        cv2.imwrite(os.path.join(self.options.output_dir, f'{event.step}.png'), img)

        return runlog


    def _read_event(self, data):
        header = struct.unpack('Q', data[:8])
        event_str = data[12:12 + int(header[0])] # 8+4
        data = data[12 + int(header[0]) + 4:]
        return data, event_str
