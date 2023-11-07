import os

from EMNIST_files.data_preparation import DataPreparation
from EMNIST_files.model import EMNIST_model
from constants import *


model = EMNIST_model(NUMBER_OF_CLASSES, MODEL_ARCHITECTURE, default_batch_size=BATCH_SIZE, default_model_path=MODEL_PATH, default_log_path=LOG_DIRECTORY)
data_preparation = DataPreparation(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES)

if os.path.exists(TRAIN_PREPARED_PATH) and os.path.exists(VALIDATION_PREPARED_PATH):
    x_train, y_train = data_preparation.load_data(TRAIN_PREPARED_PATH)
    x_validation, y_validation = data_preparation.load_data(VALIDATION_PREPARED_PATH)
    x_train, y_train = data_preparation.get_percentage_of_data(x_train, y_train, 0.5, shuffle=False)
    x_validation, y_validation = data_preparation.get_percentage_of_data(x_validation, y_validation, 0.5, shuffle=False)
    model.train(x_train, y_train, x_validation, y_validation, max_epochs_number=100, stop_after_no_improvement=5,
                learning_rate=0.001, verbose=True, validation_metric_accuracy=True)
else:
    print("Prepared data not found, execute prepare_training_and_validation_data.py first")