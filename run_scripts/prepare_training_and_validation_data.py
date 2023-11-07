from constants import *
from EMNIST_classes.data_preparation import DataPreparation

# data preparation
data_preparation = DataPreparation(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES)

x_train, y_train = data_preparation.prepare_data(X_TRAIN_PATH, Y_TRAIN_PATH)
x_validation, y_validation = data_preparation.prepare_data(X_VALIDATION_PATH, Y_VALIDATION_PATH)
#    x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
#                             rotation_angle_range_degrees=20, shift_range_percent=0.15, scale_change_range_percentage=0.2)
x_train, y_train = data_preparation.modify_data_training(x_train, y_train, equalise_classes=True, percent_create_new_examples=0.0,
                            rotation_angle_range_degrees=0.0, shift_range_percent=0.0, scale_change_range_percentage=0.0)

data_preparation.save_data(TRAIN_PREPARED_PATH, x_train, y_train)
data_preparation.save_data(VALIDATION_PREPARED_PATH, x_validation, y_validation)

print("Prepared data, saved to files")
