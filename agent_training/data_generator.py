# code adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
from tensorflow import keras
from agent_training.data_preprocessing import DataNormalizer
from agent_training.parameters import Parameters
import cv2
import os
import tensorflow as tf
from typing import List, Tuple


class DataGenerator(keras.utils.Sequence):
    """
    Generator that reads img paths, reshapes it and feeds it in batches to the neural network when training. To be used
    when dataset is very large.
    """
    def __init__(self, data_flag: str, shuffle: bool = True):
        """
        class constructor
        :param data_flag: flag to make a generator use training data or validation data
        :param shuffle: condition to shuffle the data between each epoch step when training
        """

        self.data_flag = data_flag

        self.params = Parameters()
        self.batch_size = self.params.batch_size
        self.data_path = self.params.data_path
        self.color_channels = self.params.channel_size
        self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction

        self.data_normalizer = DataNormalizer(data_path=self.data_path)
        self.list_IDs = self.data_normalizer.image_paths
        self.load_data_labels()

        if self.data_path == "validation":
            validation_size = int(len(self.list_IDs) * self.val_fraction)
            self.list_IDs = self.list_IDs[-validation_size:]
            self.labels = self.labels[-validation_size:]

        if self.data_path == "training":
            training_size = int(len(self.list_IDs) * (1 - self.val_fraction))
            self.list_IDs = self.list_IDs[:training_size]
            self.labels = self.labels[:training_size]

        self.reshape_ids_and_labels()
        self.data_size = len(self.list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, List]:
        """
        Gets next batch of data and labels
        :param index: index of which data sample to choose
        :return: training, data and its sample
        """
        if self.data_flag == 'training':
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
            # Find list of IDs
            list_ids_temp = [self.list_IDs[k] for k in indexes]
            y = self.labels[indexes]
        if self.data_flag == 'validation':
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
            # Find list of IDs
            list_ids_temp = [self.list_IDs[k] for k in indexes]
            y = self.labels[indexes]

        # Generate data
        x = self.__data_generation(list_ids_temp)
        y = self.separate_labels(y)
        print("Loading data")

        return x, y

    def on_epoch_end(self) -> None:
        """
        Shuffles the indexes of the data (to pick new random data samples)
        :return: None
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def reshape_ids_and_labels(self) -> None:
        """
        Reshapes the image_paths and the labels to have time_steps
        :return: None
        """

        if self.time_steps != 0:

            self.list_IDs = self.list_IDs[:self.list_IDs.shape[0]
                                - (self.list_IDs.shape[0] % self.time_steps)].reshape((-1, self.time_steps))
            self.labels = self.labels[: self.labels.shape[0] - (self.labels.shape[0]
                                                       % self.time_steps)].reshape((-1, self.time_steps,
                                                                                    self.labels.shape[-1]))

    def __data_generation(self, list_IDs_temp: List) -> np.ndarray:
        """
        loads a data sample
        :param list_IDs_temp: list of image-paths of which images to load
        :return: numpy array of data sample
        """
        x = []
        for ID in list_IDs_temp:
            x.append(self.load_images(ID))

        return np.array(x)

    def separate_labels(self, y: np.ndarray) -> List:
        """
        Separates the split labels in the list form that the model requires
        :return: None
        """
        if self.time_steps > 0:
            y = [y[:, :, -self.clicks_len:], y[:, :, 0:self.mouse_x_len],
                 y[:, :, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
        else:
            y = [y[:, -self.clicks_len:], y[:, 0:self.mouse_x_len],
                 y[:, self.mouse_x_len:self.mouse_x_len+self.mouse_y_len]]
        return y

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        preprocesses the image to make it in a form to input to the model
        :param image: image to be processed
        :return: processed image
        """
        image = image / 255.
        image = cv2.resize(image, self.image_size)
        return image

    def load_images(self, image_ids: List) -> List:
        """
        Loads all the preprocessed images from the image path, preprocesses them and stores them in a numpy array
        :return: None
        """
        images = []
        for image_path in image_ids:
            if '.jpg' in image_path:
                # print(os.path.join(self.data_path, image_path))
                images.append(self.preprocess_image(self.get_image(os.path.join(self.data_path, image_path.split('/')[-1]))))
        return images

    def get_image(self, img_path: str) -> np.ndarray:
        """
        Loads image to an array using keras
        :param img_path: string path to the image to be loaded
        :return: array of image
        """
        image = tf.keras.preprocessing.image.load_img(img_path, color_mode="rgb", target_size=self.image_size,
                                                      interpolation="lanczos")
        image = tf.keras.preprocessing.image.img_to_array(image, data_format=None)
        return image

    def load_data_labels(self) -> None:
        """
        loads labels from DataNormalizer object
        :return: None
        """
        x_labels, y_labels, click_labels = self.data_normalizer.one_hot_encoding()
        self.mouse_x_len = x_labels.shape[1]
        self.mouse_y_len = y_labels.shape[1]
        self.clicks_len = click_labels.shape[1]
        self.labels = np.hstack([x_labels, y_labels, click_labels])


if __name__ == "__main__":
    data_gen = DataGenerator('training')
