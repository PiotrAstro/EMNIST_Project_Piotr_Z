# Constants
from EMNIST_classes.models_architectures import *

# image information
X_SIZE = 28
Y_SIZE = 28
CHANNEL_NUMBER = 1
NUMBER_OF_CLASSES = 26

# training information
BATCH_SIZE = 128
LEARNING_RATE = 0.003
MAX_EPOCHS_NUMBER = 100
STOP_AFTER_NO_IMPROVEMENT = 5

# model architecture
MODEL_ARCHITECTURE = ModelArchitectureConvolution_v1(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES, dropout_conv=0.2, dropout_linear=0.5)

# paths
X_TRAIN_PATH = "../Data/X_train.npy"
Y_TRAIN_PATH = "../Data/y_train.npy"
X_VALIDATION_PATH = "../Data/X_val.npy"
Y_VALIDATION_PATH = "../Data/y_val.npy"
X_TEST_PATH = "../Data/X_test.npy"

Y_TEST_SAVE_PATH = "../Data/y_test.csv"

TRAIN_PREPARED_PATH = "../Data/train_prepared.npz"
VALIDATION_PREPARED_PATH = "../Data/validation_prepared.npz"

MODEL_PATH = "../models/EMNIST_model.pt"
LOG_DIRECTORY = "../models/EMNIST_log.csv"

CONFUSION_MATRIX_PATH = "../models/EMNIST_confusion_matrix.png"
ACCURACY_PLOT_PATH = "../models/EMNIST_accuracy_plot.png"
LOSS_PLOT_PATH = "../models/EMNIST_loss_plot.png"