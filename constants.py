# Constants
from EMNIST_files.models_architectures import ModelArchitectureConvolution_v1

TRAIN_MODEL = True
LOAD_PREPARED_DATA = True
X_SIZE = 28
Y_SIZE = 28
CHANNEL_NUMBER = 1
NUMBER_OF_CLASSES = 26
BATCH_SIZE = 32
MODEL_ARCHITECTURE = ModelArchitectureConvolution_v1(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES, dropout_conv=0.1, dropout_linear=0.5)
X_TRAIN_PATH = "Data/X_train.npy"
Y_TRAIN_PATH = "Data/Y_train.npy"
X_VALIDATION_PATH = "Data/X_val.npy"
Y_VALIDATION_PATH = "Data/Y_val.npy"
X_TEST_PATH = "Data/X_test.npy"
TRAIN_PREPARED_PATH = "Data/train_prepared.npz"
VALIDATION_PREPARED_PATH = "Data/validation_prepared.npz"
MODEL_PATH = "models/EMNIST_model.pt"
LOG_DIRECTORY = "models/EMNIST_log.csv"