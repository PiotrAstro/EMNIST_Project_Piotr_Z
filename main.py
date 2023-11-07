from EMNIST_files.data_preparation import DataPreparation
from EMNIST_files.model import EMNIST_model
from EMNIST_files.models_architectures import ModelArchitectureConvolution_v1, ModelArchitectureLinear_v1

import os

# Constants
LOAD_PREPARED_DATA = True
X_SIZE = 28
Y_SIZE = 28
CHANNEL_NUMBER = 1
NUMBER_OF_CLASSES = 26
X_TRAIN_PATH = "Data/X_train.npy"
Y_TRAIN_PATH = "Data/Y_train.npy"
X_VALIDATION_PATH = "Data/X_val.npy"
Y_VALIDATION_PATH = "Data/Y_val.npy"
TRAIN_PREPARED_PATH = "Data/train_prepared.npz"
VALIDATION_PREPARED_PATH = "Data/validation_prepared.npz"
MODEL_PATH = "models/EMNIST_model.pt"

# Create model
#model_architecture = ModelArchitectureConvolution_v1(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES, dropout_conv=0.25, dropout_linear=0.5)
model_architecture = ModelArchitectureConvolution_v1(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES, dropout_linear=0.0)
model = EMNIST_model(NUMBER_OF_CLASSES, model_architecture, default_batch_size=64, default_model_path=MODEL_PATH)

# data preparation
data_preparation = DataPreparation(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES)

if(LOAD_PREPARED_DATA and os.path.exists(TRAIN_PREPARED_PATH) and os.path.exists(VALIDATION_PREPARED_PATH)):
    x_train, y_train = data_preparation.load_data(TRAIN_PREPARED_PATH)
    x_validation, y_validation = data_preparation.load_data(VALIDATION_PREPARED_PATH)
    x_train, y_train = data_preparation.get_percentage_of_data(x_train, y_train, 0.1)
    x_validation, y_validation = data_preparation.get_percentage_of_data(x_validation, y_validation, 0.1)

    print("Loaded prepared data")
else:
    x_train, y_train = data_preparation.prepare_data(X_TRAIN_PATH, Y_TRAIN_PATH)
    x_validation, y_validation = data_preparation.prepare_data(X_VALIDATION_PATH, Y_VALIDATION_PATH)
#    x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
#                             rotation_angle_range_degrees=20, shift_range_percent=0.15, scale_change_range_percentage=0.2)
    x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
                                rotation_angle_range_degrees=0.0, shift_range_percent=0.0, scale_change_range_percentage=0.0)

    data_preparation.save_data(TRAIN_PREPARED_PATH, x_train, y_train)
    data_preparation.save_data(VALIDATION_PREPARED_PATH, x_validation, y_validation)

    print("Prepared data, saved to files")

# train model
model.train(x_train, y_train, x_validation, y_validation, max_epochs_number=100, stop_after_no_improvement=5, learning_rate=0.001, verbose=True, validation_metric_accuracy=True)

