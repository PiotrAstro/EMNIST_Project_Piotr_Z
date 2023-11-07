import os

from EMNIST_files.data_preparation import DataPreparation
from EMNIST_files.model import EMNIST_model
from EMNIST_files.result_visualisation import draw_confusion_matrix, draw_plot_accuracy, draw_plot_loss
from constants import *

model = EMNIST_model(NUMBER_OF_CLASSES, MODEL_ARCHITECTURE, default_batch_size=BATCH_SIZE, default_model_path=MODEL_PATH, default_log_path=LOG_DIRECTORY)
data_preparation = DataPreparation(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES)
model.load_weights(MODEL_PATH)

if os.path.exists(TRAIN_PREPARED_PATH) and os.path.exists(VALIDATION_PREPARED_PATH):
    x_validation, y_validation = data_preparation.load_data(VALIDATION_PREPARED_PATH)
    x_test = data_preparation.prepare_x(X_TEST_PATH)

# test model
predicted_validation = model.predict(x_validation)
draw_confusion_matrix(y_validation, predicted_validation)
training_accuracy, validation_accuracy, training_loss, validation_loss, saves = model.load_logs_accuracy_losses_saves(LOG_DIRECTORY)
draw_plot_accuracy(training_accuracy, validation_accuracy, saves)
draw_plot_loss(training_loss, validation_loss, saves)
