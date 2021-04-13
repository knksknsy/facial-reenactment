from enum import Enum

class Mode(Enum):
    DATASET = 'dataset'
    TRAIN = 'train'
    TEST = 'test'
    INFER = 'infer'
    LOGS = 'logs'


class Method(Enum):
    CREATION = 'creation'
    DETECTION = 'detection'


def get_progress(batch_num: int, len_data_loader: int, limit: int = None):
    counter = str(batch_num + 1).zfill(len(str(len_data_loader)))
    denominator = len_data_loader if limit is None else limit
    return f'[{counter}/{denominator}]'
