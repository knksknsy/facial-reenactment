from configs import config, Options

class TrainOptions(Options):
    def __init__(self, description):
        super(TrainOptions, self).__init__(description)
        self._init_parser()
        self.args = self._parse_args()
        self.losses = self._parse_losses(self.args)


    def _init_parser(self):
        self.parser.add_argument('--pin_memory', action='store_true')
        self.parser.add_argument('--num_workers', type=int, default=0)
        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--dataset_train', type=str, required=True,
                                            help='Path to the pre-processed dataset for training.')
        self.parser.add_argument('--dataset_test', type=str, required=True,
                                            help='Path to the pre-processed dataset for testing.')
        self.parser.add_argument('--csv_train', type=str, required=True,
                                            help='Path to CSV file needed for torch.utils.data.Dataset to load data for training.')
        self.parser.add_argument('--csv_test', type=str, required=True,
                                            help='Path to CSV file needed for torch.utils.data.Dataset to load data for testing.')
        self.parser.add_argument('--log_dir', type=str, default=config.LOG_DIR,
                                            help='Path where logs will be saved.')
        self.parser.add_argument('--models_dir', type=str, default=config.MODELS_DIR,
                                            help='Path where models will be saved.')
        self.parser.add_argument('--gen_dir', type=str, default=config.GENERATED_DIR,
                                            help='Path where generated images will be saved.')
        self.parser.add_argument('--gen_test_dir', type=str, default=config.GENERATED_DIR,
                                            help='Path where generated test images will be saved.')

        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', nargs='?', default='cuda', const='cuda', choices=['cuda', 'cpu'],
                                            help='Whether to run the model on GPU or CPU.')
        self.parser.add_argument('--continue_id', type=str, default=None,
                                            help='Id of the models to continue training.')
        self.parser.add_argument('--image_size', default=config.IMAGE_SIZE, type=int,
                                            help='Image size')
        self.parser.add_argument('--angle', type=int, default=config.ROTATION_ANGLE,
                                            help='Angle for random image rotation when loading data.')

        # ARGUMENTS: LOSS WEIGHT
        self.parser.add_argument('--l_adv', default=config.LOSS_ADV, type=float,
                                            help='Adversarial loss')
        self.parser.add_argument('--l_rec', default=config.LOSS_REC, type=float,
                                            help='Reconstruction loss')
        self.parser.add_argument('--l_self', default=config.LOSS_SELF, type=float,
                                            help='Cycle consistency loss')
        self.parser.add_argument('--l_triple', default=config.LOSS_TRIPLE, type=float,
                                            help='Triple consistency loss')
        self.parser.add_argument('--l_percep', default=config.LOSS_PERCEP, type=float,
                                            help='Perceptual loss')
        self.parser.add_argument('--l_tv', default=config.LOSS_TV, type=float,
                                            help='Total variation loss')

        # ARGUMENTS: HYPERPARAMETERS
        self.parser.add_argument('--batch_size', default=config.BATCH_SIZE, type=int,
                                            help='Batch size')
        self.parser.add_argument('--epochs', default=config.EPOCHS, type=int,
                                            help='Epochs to train')
        self.parser.add_argument('--lr_g', default=config.LEARNING_RATE_G, type=float,
                                            help='Learning rate of generator')
        self.parser.add_argument('--lr_d', default=config.LEARNING_RATE_D, type=float,
                                            help='Learning rate of discriminator')
        self.parser.add_argument('--grad_clip', default=config.GRADIENT_CLIPPING, type=float,
                                            help='Use gradient clipping')
        self.parser.add_argument('--gan_type', default=config.GAN_TYPE, type=str,
                                            help='GAN type')

        # ARGUMENTS: OPTIMIZER
        self.parser.add_argument('--beta1', default=config.BETA1, type=float,
                                            help='Beta1 of Adam optimizer')
        self.parser.add_argument('--beta2', default=config.BETA2, type=float,
                                            help='Beta2 of Adam optimizer')
        self.parser.add_argument('--weight_decay', default=config.WEIGHT_DECAY, type=float,
                                            help='Weight decay of optimizer')
        self.parser.add_argument('--scheduler_step_size', default=config.SCHEDULER_STEP_SIZE, type=int,
                                            help='Step size for scheduler')
        self.parser.add_argument('--scheduler_gamma', default=config.SCHEDULER_GAMMA, type=float,
                                            help='Gamma for scheduler')


    def _parse_losses(self, args):
        loss_keys = [k for k in list(args.__dict__.keys()) if 'l_' in k]
        losses = dict(zip([k.replace('l_', '') for k in loss_keys], [float(args.__getattribute__(v)) for v in loss_keys]))
        return losses
