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
