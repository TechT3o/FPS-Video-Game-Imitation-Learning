# code adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
from tensorflow import keras
from agent_training.data_preprocessing import DataProcessor
from sklearn.model_selection import train_test_split
import cv2
import os
import tensorflow as tf


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_flag, shuffle=True):

        self.data_flag = data_flag
        self.data_processor = DataProcessor()
        self.data_path = self.data_processor.data_path
        self.image_size = self.data_processor.image_size
        self.time_steps = self.data_processor.time_steps
        self.val_fraction = self.data_processor.val_fraction
        self.test_fraction = self.data_processor.test_fraction
        self.batch_size = self.data_processor.batch_size

        self.color_channels = self.data_processor.color_channels
        self.list_IDs = self.data_processor.image_paths
        self.labels = self.data_processor.labels
        self.reshape_ids_and_labels()

        self.shuffle = shuffle
        self.on_epoch_end()
        self.train_test_val_split()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        if self.data_flag == 'train':
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            y = self.labels[indexes]
        if self.data_flag == 'validation':
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            y = self.labels[indexes]

        # Generate data
        # print(len(list_IDs_temp))
        X = self.__data_generation(list_IDs_temp)
        # print(f'data_shape {X.shape}')
        y = self.seperate_labels(y)
        # print(f'labels are {y}')

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def reshape_ids_and_labels(self):
        if self.time_steps != 0:

            self.list_IDs = self.list_IDs[:self.list_IDs.shape[0]
                                - (self.list_IDs.shape[0] % self.time_steps)].reshape((-1, self.time_steps))
            self.labels = self.labels[: self.labels.shape[0] - (self.labels.shape[0]
                                                       % self.time_steps)].reshape((-1, self.time_steps,
                                                                                    self.labels.shape[-1]))

    def __data_generation(self, list_IDs_temp):

        x = []
        for ID in list_IDs_temp:
            x.append(self.load_images(ID))

        return np.array(x)

    def separate_labels_for_multiple_out(self):
        if self.time_steps > 0:
            self.__y_train = [self.__y_train[:, :, -self.clicks_len:], self.__y_train[:, :, 0:self.mouse_x_len],
                              self.__y_train[:, :, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
            self.__y_test = [self.__y_test[:, :, -self.clicks_len:], self.__y_test[:, :, 0:self.mouse_x_len],
                             self.__y_test[:, :, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
            self.__y_val = [self.__y_val[:, :, -self.clicks_len:], self.__y_val[:, :, 0:self.mouse_x_len],
                            self.__y_val[:, :, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
        else:
            self.__y_train = [self.__y_train[:, -self.clicks_len:], self.__y_train[:, 0:self.mouse_x_len],
                              self.__y_train[:, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
            self.__y_test = [self.__y_test[:, -self.clicks_len:], self.__y_test[:, 0:self.mouse_x_len],
                             self.__y_test[:, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
            self.__y_val = [self.__y_val[:, -self.clicks_len:], self.__y_val[:, 0:self.mouse_x_len],
                            self.__y_val[:, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]

    def seperate_labels(self, y):
        if self.time_steps > 0:
            y = [y[:, :, -self.data_processor.clicks_len:], y[:, :, 0:self.data_processor.mouse_x_len],
                 y[:, :, self.data_processor.mouse_x_len:self.data_processor.mouse_x_len+self.data_processor.mouse_y_len]]
        else:
            y = [y[:, -self.data_processor.clicks_len:], y[:, 0:self.data_processor.mouse_x_len],
                 y[:, self.data_processor.mouse_x_len:self.data_processor.mouse_x_len+self.data_processor.mouse_y_len]]
        return y

    def preprocess_image(self, image):
        image = image / 255.
        image = cv2.resize(image, self.image_size)
        return image

    def load_images(self, image_ids):
        images = []
        for image_path in image_ids:
            if '.jpg' in image_path:
                # print(os.path.join(self.data_path, image_path))
                images.append(self.preprocess_image(self.get_image(os.path.join(self.data_path, image_path))))
        return images

    def get_image(self, img_path) -> np.ndarray:
        """
        Loads image to an array using keras
        :param img_path: string path to the image to be loaded
        :return: array of image
        """
        image = tf.keras.preprocessing.image.load_img(img_path, color_mode="rgb", target_size=self.image_size,
                                                      interpolation="lanczos")
        image = tf.keras.preprocessing.image.img_to_array(image, data_format=None)
        return image


if __name__ == "__main__":
    data_gen = DataGenerator('train')
