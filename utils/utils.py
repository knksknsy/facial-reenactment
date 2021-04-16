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


def add_losses(acc_dict, new_dict, batch_size: int = None):
        for k, v in new_dict.items():
            acc_dict[k] += new_dict[k] if batch_size is None else batch_size * new_dict[k]
        return acc_dict


def avg_losses(losses_dict, iterations: int = None):
    for k, v in losses_dict.items():
        losses_dict[k] = v / iterations
    return losses_dict


def init_feature_losses():
    losses = dict({
        'Loss_Contr': 0.0,
        'Loss_Contr_Real': 0.0,
        'Loss_Contr_Fake': 0.0,
        'Loss_Mask': 0.0,
        'Loss_Feature': 0.0
    })
    return losses


def init_class_losses():
    losses = dict({
        'Loss_BCE': 0.0,
        'Loss_Mask': 0.0,
        'Loss_Class': 0.0
    })
    return losses
