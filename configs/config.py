# PATHS -----------------------------------------------------
LOG_DIR = r'logs'
MODELS_DIR = r'models'
GENERATED_DIR = r'generated_img'
GENERATED_TEST_DIR = r'generated_test'
TEST_DIR = r'outputs'
# VGG_FACE = r'/home/<user>/models/vgg_face_dag.pth'

# TESTING
IMAGE_SIZE_TEST = 299

# DATASET ---------------------------------------------------
K = 8

# TRAINING --------------------------------------------------
ROTATION_ANGLE = 15

# Training hyperparameters
IMAGE_SIZE = 128  # 224
BATCH_SIZE = 8
EPOCHS = 30

LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0001

LOSS_ADV = 0.01
LOSS_REC = 1
LOSS_SELF = 1
LOSS_TRIPLE = 1
LOSS_PERCEP = 0.1
LOSS_TV = 1e-8

# Model parameter
GAN_TYPE = 'wgan-gp'
GRADIENT_CLIPPING = 1

# Optimizer parameter
BETA1 = 0.5
BETA2 = 0.9
WEIGHT_DECAY = 5*1e-4
SCHEDULER_STEP_SIZE = 1
SCHEDULER_GAMMA = 0.1
