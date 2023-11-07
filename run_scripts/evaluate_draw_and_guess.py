import os
import random

from EMNIST_classes.data_preparation import DataPreparation
from EMNIST_classes.model import EMNIST_model
from EMNIST_classes.result_visualisation import draw_confusion_matrix, draw_plot_accuracy, draw_plot_loss
from constants import *




def class_number_to_char(class_number):
    return chr(class_number + 65)

model = EMNIST_model(NUMBER_OF_CLASSES, MODEL_ARCHITECTURE, default_batch_size=BATCH_SIZE, default_model_path=MODEL_PATH, default_log_path=LOG_DIRECTORY)
data_preparation = DataPreparation(CHANNEL_NUMBER, X_SIZE, Y_SIZE, NUMBER_OF_CLASSES)
model.load_weights(MODEL_PATH)

if os.path.exists(TRAIN_PREPARED_PATH) and os.path.exists(VALIDATION_PREPARED_PATH):
    x_validation, y_validation = data_preparation.load_data(VALIDATION_PREPARED_PATH)

    while True:
        print("Press Enter to continue to the next iteration or Backspace to stop.")

        # Wait for user input
        user_input = input()

        # Check if the user wants to continue or stop
        if user_input == '':  # User pressed Enter
            print("Next iteration...")
            random_index = random.randint(0, len(x_validation) - 1)
            x = x_validation[random_index]
            y = y_validation[random_index]
            predicted_class = model.predict_sinle_input(x)
            print(f"\n\n\nIndex of image: {random_index}")
            data_preparation.print_console_image(x)
            print(f"Predicted class: {class_number_to_char(predicted_class)}")
            print(f"Correct class: {class_number_to_char(y)}")

        elif user_input == '\x08':  # User pressed Backspace (may not work in all terminals)
            print("Stopping...")
            break
        else:
            print("Invalid input. Try again.")
