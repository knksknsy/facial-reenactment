from loggings.logger import Logger
from configs.options import Options

from models.detection_network import NetworkDetection

class InferDetection():

    def __init__(self, logger: Logger, options: Options, model_path: str):
        self.logger = logger
        self.options = options
        self.model_path = model_path

        self.network = NetworkDetection(self.logger, self.options, self.model_path)
        self.network.eval()
