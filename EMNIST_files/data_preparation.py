from random import random
from scipy.ndimage import rotate
import numpy as np


class DataPreparation:
    def __init__(self, channels_number, x_size, y_size, number_of_classes, max_pixel_value=255):
        self.channels_number = channels_number
        self.x_size = x_size
        self.y_size = y_size
        self.number_of_classes = number_of_classes
        self.max_pixel_value = max_pixel_value

    def prepare_x(self, data_path):
        data = np.load(data_path)
        data = data.astype(np.float32)
        data /= self.max_pixel_value
        data = np.reshape(data, (-1, self.channels_number, self.x_size, self.y_size))
        return data

    def prepare_y(self, labels_path):
        labels = np.load(labels_path)
        labels = labels.astype(np.long)
        labels = np.max(labels, axis=1)
        return labels

    def prepare_data(self, data_path, labels_path):
        x = self.load_x(data_path)
        y = self.load_y(labels_path)
        return x, y

    def modify_data_training(self, x_prepared, y_prepared, equalise_classes=True, percent_create_new_examples=0.3,
                             rotation_angle_range_degrees=20, shift_range_percent=0.2, scale_change_range_percentage=0.3):
        x_by_classes = [[] for _ in range(self.number_of_classes)]
        y_by_classes = [[] for _ in range(self.number_of_classes)]
        original_length_of_classes = [0 for _ in range(self.number_of_classes)]
        goal_length_of_classes = [0 for _ in range(self.number_of_classes)]

        for i in range(len(x_prepared)):
            x_by_classes[y_prepared[i]].append(x_prepared[i])
            y_by_classes[y_prepared[i]].append(y_prepared[i])

        original_length_of_classes = [len(x_by_classes[i]) for i in range(self.number_of_classes)]
        if equalise_classes:
            max_class_size = max([len(x) for x in x_by_classes])
            for i in range(self.number_of_classes):
                goal_length_of_classes[i] = max_class_size * int(1 + percent_create_new_examples)
        else:
            goal_length_of_classes = [len(x_by_classes[i]) * int(1 + percent_create_new_examples) for i in range(self.number_of_classes)]

        for i in range(self.number_of_classes):
            while len(x_by_classes[i]) < goal_length_of_classes[i]:
                add_based_on_index = random.randint(0, original_length_of_classes[i] - 1)
                x_by_classes[i].append(self.create_new_example(x_by_classes[i][add_based_on_index], rotation_angle_range_degrees, shift_range_percent, scale_change_range_percentage))
                y_by_classes[i].append(y_by_classes[i][add_based_on_index])

    def create_new_example(self, image, rotation_angle_range_degrees, shift_range_percent, scale_change_range_percentage):
        rotate_angle = random.uniform(-rotation_angle_range_degrees, rotation_angle_range_degrees)
        shift_x = random.uniform(-shift_range_percent, shift_range_percent)
        shift_y = random.uniform(-shift_range_percent, shift_range_percent)
        scale_change = random.uniform(-scale_change_range_percentage, scale_change_range_percentage)
        image = image = np.transpose(image, (2, 1, 0))
        image = rotate(image, angle=rotation_angle_range_degrees, axes=(1, 0), reshape=False, mode='nearest')
        image = 

        image = np.transpose(image, (2, 1, 0))
        return image


