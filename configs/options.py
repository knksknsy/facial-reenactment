import argparse
import torch

from configs import config

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Facial Reenactment')
        self._init_subparsers()
        self.losses
        self.optimizer
        self.device
        self.args = self._parse_args()


    def _init_subparsers(self):
        self.subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # ARGUMENTS: DATASET PRE-PROCESSING ---------------------------------------------------------------------------------
        self.dataset_parser = self.subparsers.add_parser("dataset", help="Pre-process the dataset for its use.")
        self.dataset_parser.add_argument("--source", type=str, required=True,
                                            help="Path to the source folder where the raw VoxCeleb2 dataset is located.")
        self.dataset_parser.add_argument("--output", type=str, required=True,
                                            help="Path to the folder where the pre-processed dataset will be stored.")
        self.dataset_parser.add_argument("--csv", type=str, required=True,
                                            help="Path to where the CSV file will be saved.")
        self.dataset_parser.add_argument("--frames", type=int, default=config.K+1,
                                            help="Number of frames + 1 to extract from a video.")
        self.dataset_parser.add_argument("--size", type=int, default=0,
                                            help="Number of videos from the dataset to process. Providing 0 will pre-process all videos.")
        self.dataset_parser.add_argument("--gpu", action="store_true",
                                            help="Run the model (face_alignment) on GPU.")
        self.dataset_parser.add_argument("--overwrite", action="store_true",
                                            help="Add this flag to overwrite already pre-processed files. The default functionality"
                                            "is to ignore videos that have already been pre-processed.")
        self.dataset_parser.add_argument("--log_dir", type=str, default=config.LOG_DIR,
                                            help="Path where logs will be saved.")
        
        # ARGUMENTS: TRAINING ------------------------------------------------------------------------------------------------
        self.train_parser = self.subparsers.add_parser("train", help="Starts the training process.")
        self.train_parser.add_argument("--dataset_train", type=str, required=True,
                                            help="Path to the pre-processed dataset for training.")
        self.train_parser.add_argument("--dataset_test", type=str, required=True,
                                            help="Path to the pre-processed dataset for testing.")
        self.train_parser.add_argument("--csv_train", type=str, required=True,
                                            help="Path to CSV file needed for torch.utils.data.Dataset to load data for training.")
        self.train_parser.add_argument("--csv_test", type=str, required=True,
                                            help="Path to CSV file needed for torch.utils.data.Dataset to load data for testing.")
        self.train_parser.add_argument("--gpu", action="store_true",
                                            help="Run the model on GPU.")
        self.train_parser.add_argument("--continue_id", type=str, default=None,
                                            help="Id of the models to continue training.")
        self.train_parser.add_argument("--angle", type=int, default=config.ROTATION_ANGLE,
                                            help="Angle for random image rotation when loading data.")

        # ARGUMENTS: TRAINING: DIRECTORIES -------------------------------------------------------------------------------------
        self.train_parser.add_argument("--log_dir", type=str, default=config.LOG_DIR,
                                            help="Path where logs will be saved.")
        self.train_parser.add_argument("--models_dir", type=str, default=config.MODELS_DIR,
                                            help="Path where models will be saved.")
        self.train_parser.add_argument("--gen_dir", type=str, default=config.GENERATED_DIR,
                                            help="Path where generated images will be saved.")
        
        # ARGUMENTS: TRAINING: LOSS WEIGHTS ------------------------------------------------------------------------------------
        self.train_parser.add_argument("--l_adv", default=config.LOSS_ADV, type=float,
                                            help="Adversarial loss")
        self.train_parser.add_argument("--l_rec", default=config.LOSS_REC, type=float,
                                            help="Reconstruction loss")
        self.train_parser.add_argument("--l_self", default=config.LOSS_SELF, type=float,
                                            help="Cycle consistency loss")
        self.train_parser.add_argument("--l_triple", default=config.LOSS_TRIPLE, type=float,
                                            help="Triple consistency loss")
        self.train_parser.add_argument("--l_percep", default=config.LOSS_PERCEP, type=float,
                                            help="Perceptual loss")
        self.train_parser.add_argument("--l_tv", default=config.LOSS_TV, type=float,
                                            help="Total variation loss")

        # ARGUMENTS: TRAINING: HYPERPARAMETERS ---------------------------------------------------------------------------------
        self.train_parser.add_argument("--image_size", default=config.IMAGE_SIZE, type=int,
                                            help="Image size")
        self.train_parser.add_argument("--batch_size", default=config.BATCH_SIZE, type=int,
                                            help="Batch size")
        self.train_parser.add_argument("--epochs", default=config.EPOCHS, type=int,
                                            help="Epochs to train")
        self.train_parser.add_argument("--lr_g", default=config.LEARNING_RATE_G, type=float,
                                            help="Learning rate of generator")
        self.train_parser.add_argument("--lr_d", default=config.LEARNING_RATE_D, type=float,
                                            help="Learning rate of discriminator")
        self.train_parser.add_argument("--grad_clip", default=config.GRADIENT_CLIPPING, type=float,
                                            help="Use gradient clipping")
        self.train_parser.add_argument("--gan_type", default=config.GAN_TYPE, type=str,
                                            help="GAN type")

        # ARGUMENTS: TRAINING: OPTIMIZER ---------------------------------------------------------------------------------
        self.train_parser.add_argument("--optim_type", default=config.OPTIMIZER, type=str,
                                            help="Optimizer")
        self.train_parser.add_argument("--optim_params", default=config.OPTIMIZER_PARAMETERS, nargs='+',
                                            help="List of optimizer parameters")
        self.train_parser.add_argument("--optim_weight_decay", default=config.OPTIMIZER_WEIGHT_DECAY, type=float,
                                            help="Weight decay of optimizer")
        self.train_parser.add_argument("--optim_step_size", default=config.OPTIMIZER_SCHEDULER_STEP_SIZE, type=int,
                                            help="Step size for scheduler")
        self.train_parser.add_argument("--optim_gamma", default=config.OPTIMIZER_SCHEDULER_GAMMA, type=float,
                                            help="Gamma for scheduler")


    def _parse_args(self):
        self.args = self.parser.parse_args()
        self.device = 'cuda' if (torch.cuda.is_available() and self.args.gpu) else 'cpu'
        
        if self.args.subcommand == "train":
            self.losses = self._parse_losses()
            self.optimizer = self._parse_optimizer()

        return self.args


    def _parse_losses(self):
        loss_keys = [k for k in list(self.args.__dict__.keys()) if 'l_' in k]
        self.losses = dict(zip([k.replace('l_', '') for k in loss_keys], [float(self.args.__getattribute__(v)) for v in loss_keys]))
        return self.losses


    def _parse_optimizer(self):
        optim_keys = [k for k in list(self.args.__dict__.keys()) if 'optim_' in k]
        self.optimizer = dict(zip([k.replace('optim_', '') for k in optim_keys], [float(self.args.__getattribute__(v)) for v in optim_keys]))
        return self.optimizer    
