from EMNIST_files.data_preparation import DataPreparation
from EMNIST_files.model import EMNIST_model
from EMNIST_files.models_architectures import ModelArchitectureConvolution_v1, ModelArchitectureLinear_v1

import os
import time

from EMNIST_files.result_visualisation import draw_confusion_matrix, draw_plot_accuracy, draw_plot_loss



# Create model

#model_architecture = ModelArchitectureLinear_v1(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES, dropout_linear=0.5)


# data preparation


    print("Loaded prepared data")
else:
    x_train, y_train = data_preparation.prepare_data(X_TRAIN_PATH, Y_TRAIN_PATH)
    x_validation, y_validation = data_preparation.prepare_data(X_VALIDATION_PATH, Y_VALIDATION_PATH)
    x_validation, y_validation = data_preparation.shuffle_data(x_validation, y_validation)
#    x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
#                             rotation_angle_range_degrees=20, shift_range_percent=0.15, scale_change_range_percentage=0.2)
    x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
                                rotation_angle_range_degrees=0.0, shift_range_percent=0.0, scale_change_range_percentage=0.0)

    data_preparation.save_data(TRAIN_PREPARED_PATH, x_train, y_train)
    data_preparation.save_data(VALIDATION_PREPARED_PATH, x_validation, y_validation)

    print("Prepared data, saved to files")


if TRAIN_MODEL:
    # train model
    else:
    # load model
