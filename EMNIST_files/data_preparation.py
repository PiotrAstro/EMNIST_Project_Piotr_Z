from scipy.ndimage import rotate, zoom, shift

import numpy as np
import random


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
        labels = np.argmax(labels, axis=1)
        return labels

    def prepare_data(self, data_path, labels_path):
        x = self.prepare_x(data_path)
        y = self.prepare_y(labels_path)
        return x, y

    def shuffle_data(self, x, y):
        permutation = np.random.permutation(len(x))
        return x[permutation], y[permutation]

    def modify_data_training(self, x_prepared, y_prepared, equalise_classes=True, percent_create_new_examples=0.3,
                             rotation_angle_range_degrees=20, shift_range_percent=0.2, scale_change_range_percentage=0.3):
        x_by_classes = [[] for _ in range(self.number_of_classes)]
        y_by_classes = [[] for _ in range(self.number_of_classes)]

        for i in range(len(x_prepared)):
            x_by_classes[y_prepared[i]].append(x_prepared[i])
            y_by_classes[y_prepared[i]].append(y_prepared[i])

        original_length_of_classes = [len(x_by_classes[i]) for i in range(self.number_of_classes)]
        if equalise_classes:
            max_class_size = max([len(x) for x in x_by_classes])
            goal_length_of_classes = [int(max_class_size * (1 + percent_create_new_examples))] * self.number_of_classes
        else:
            goal_length_of_classes = [int(len(x_by_classes[i]) * (1 + percent_create_new_examples)) for i in range(self.number_of_classes)]

        for i in range(self.number_of_classes):
            while len(x_by_classes[i]) < goal_length_of_classes[i]:
                add_based_on_index = random.randint(0, original_length_of_classes[i] - 1)
                x_by_classes[i].append(self.create_new_example(x_by_classes[i][add_based_on_index], rotation_angle_range_degrees, shift_range_percent, scale_change_range_percentage))
                y_by_classes[i].append(y_by_classes[i][add_based_on_index])

        x_result = []
        y_result = []
        for i in range(self.number_of_classes):
            x_result.extend(x_by_classes[i])
            y_result.extend(y_by_classes[i])
        x_result = np.array(x_result)
        y_result = np.array(y_result)

        x_result, y_result = self.shuffle_data(x_result, y_result)

        return x_result, y_result

    def create_new_example(self, image, rotation_angle_range_degrees, shift_range_percent, scale_change_range_percentage):
        rotate_angle = random.uniform(-rotation_angle_range_degrees, rotation_angle_range_degrees)
        shift_x = random.uniform(-shift_range_percent * self.x_size, shift_range_percent * self.x_size)
        shift_y = random.uniform(-shift_range_percent * self.y_size, shift_range_percent * self.y_size)
        scale_change = 1 + random.uniform(-scale_change_range_percentage, scale_change_range_percentage)

        image = np.transpose(image, (2, 1, 0))
        if scale_change_range_percentage > 0 and scale_change != 1:
            image = self.resize_and_maintain_size(image, scale_change)
        if rotation_angle_range_degrees > 0 and rotate_angle != 0:
            image = self.rotate_and_maintain_size(image, rotate_angle)
        if shift_range_percent > 0 and (shift_x != 0 or shift_y != 0):
            image = self.shift_and_maintain_size(image, shift_x, shift_y)
        image = np.transpose(image, (2, 1, 0))
        return image

    def save_data(self, path,  x, y=np.array([])):
        np.savez(path, x=x, y=y)

    def load_data(self, path):
        data = np.load(path)
        x = data['x']
        y = data['y']
        return x, y

    def print_console_image(self, image):
        image_flattened = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        for i in range(self.channels_number):
            image_flattened += image[i]
        image_flattened /= self.channels_number
        print_text = ''

        for row in range(self.y_size):
            for column in range(self.x_size):
                if image_flattened[row][column] > 0.8:
                    print_text += '0'
                elif image_flattened[row][column] > 0.4:
                    print_text += 'o'
                elif image_flattened[row][column] > 0.2:
                    print_text += '.'
                else:
                    print_text += ' '
            print_text += '\n'
        print(print_text)

    def shift_and_maintain_size(self, image, shift_x, shift_y): # image in (height, width, channels) format
        return shift(image, (shift_y, shift_x, 0), order=3, mode='nearest')

    def rotate_and_maintain_size(self, image, angle): # image in (height, width, channels) format
        return rotate(image, angle=angle, axes=(1, 0), reshape=False, mode='nearest')

    def resize_and_maintain_size(self, image, scale_factor):
        # Use scipy.ndimage.zoom to scale the image
        zoomed_image = zoom(image, (scale_factor, scale_factor, 1), order=3)

        # Crop or pad to maintain the original size
        if scale_factor > 1:  # Crop
            start_row = int((zoomed_image.shape[0] - self.y_size) / 2)
            start_col = int((zoomed_image.shape[1] - self.x_size) / 2)
            zoomed_image = zoomed_image[start_row:start_row + self.y_size, start_col:start_col + self.x_size]
        elif scale_factor < 1:  # Pad
            padding_top = int((self.y_size - zoomed_image.shape[0]) / 2)
            padding_bottom = self.y_size - zoomed_image.shape[0] - padding_top
            padding_left = int((self.x_size - zoomed_image.shape[1]) / 2)
            padding_right = self.x_size - zoomed_image.shape[1] - padding_left
            zoomed_image = np.pad(zoomed_image, ((padding_top, padding_bottom),
                                                 (padding_left, padding_right),
                                                 (0, 0)),
                                  mode='edge')
        return zoomed_image

    def get_percentage_of_data(self, x, y, percentage, shuffle=True):
        if percentage >= 1:
            return x, y
        if shuffle:
            x_result = []
            y_result = []
            for i in range(len(x)):
                if random.uniform(0, 1) < percentage:
                    x_result.append(x[i])
                    y_result.append(y[i])
            return np.array(x_result), np.array(y_result)
        else:
            return x[:int(len(x) * percentage)], y[:int(len(y) * percentage)]