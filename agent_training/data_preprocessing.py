import cv2
from sklearn.model_selection import train_test_split
from agent_training.data_normalizer import DataNormalizer
from agent_training.parameters import Parameters
import tensorflow as tf
from typing import Tuple, List
import numpy as np
import os


class DataProcessor:
    """
    Class that reads the stored data, preprocesses them and splits them into training test and validation for the
    model trainer to use.
    """
    data_path: str
    save_path: str
    batch_size: int
    val_fraction: float
    test_fraction: float
    image_size: Tuple[int, int]
    time_steps: int
    color_channels: int
    normalize: bool
    __X: np.ndarray
    __y: np.ndarray
    __X_train: np.ndarray
    __y_train: np.ndarray or List
    __X_val: np.ndarray
    __y_val: np.ndarray or List
    __X_test: np.ndarray
    __y_test: np.ndarray or List

    def __init__(self):

        self.params = Parameters()
        self.data_path = self.params.data_path
        self.color_channels = self.params.channel_size
        self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction
        self.test_fraction = self.params.test_fraction

        self.__label_indices = dict()
        self.data_normalizer = DataNormalizer(data_path=self.data_path)
        self.image_paths = self.data_normalizer.image_paths
        self.prepare_data()

    def prepare_data(self):
        self.load_data_labels()
        self.load_images()
        self.reshape_dataset()
        self.train_test_val_split()
        self.separate_labels_for_multiple_out()

    def load_data_labels(self) -> None:

        x_labels, y_labels, click_labels = self.data_normalizer.one_hot_encoding()
        print(x_labels.shape)
        self.mouse_x_len = x_labels.shape[1]
        self.mouse_y_len = y_labels.shape[1]
        self.clicks_len = click_labels.shape[1]
        print(self.mouse_x_len, self.mouse_y_len, self.clicks_len)
        self.__y = np.hstack([x_labels, y_labels, click_labels])

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

    def reshape_dataset(self) -> None:
        """
        Reshape the dataset to be loaded in an R-CNN
        :return: None
        """

        # if self.time_steps != 0 and len(self.__y.shape) == 1:
        #     self.__X = self.__X.reshape((-1, self.time_steps, self.image_size[0],
        #                                 self.image_size[1], self.color_channels))
        #     self.__y = self.__y.reshape(-1, self.time_steps)
        #     self.__y = self.__y = np.mean(self.__y, axis=1)
        if self.time_steps != 0:
            # print(self.__X.shape)
            self.__X = self.__X[:self.__X.shape[0]
                                - (self.__X.shape[0] % self.time_steps)].reshape((-1, self.time_steps,
                                                                                  self.image_size[0],
                                                                                  self.image_size[1],
                                                                                  self.color_channels))
            self.__y = self.__y[: self.__y.shape[0] - (self.__y.shape[0]
                                                       % self.time_steps)].reshape((-1,
                                                                                    self.time_steps,
                                                                                    self.__y.shape[-1]))
        if self.time_steps == 0:
            self.__X = np.swapaxes(self.__X, 1, 2)

    def preprocess_image(self, image):
        image = image / 255.
        image = cv2.resize(image, self.image_size)
        return image

    def load_images(self):
        x = []
        for image_path in self.image_paths:
            if '.jpg' in image_path:
                processed_image = self.preprocess_image(self.get_image(os.path.join(self.data_path, image_path)))
                x.append(processed_image)
        self.__X = np.array(x)

    def train_test_val_split(self) -> None:
        """
        Splits data in train, test and validation
        :return: None
        """

        test_fraction = self.test_fraction / (1 - self.val_fraction)

        self.__X_train, self.__X_val, self.__y_train, self.__y_val = train_test_split(self.__X, self.__y,
                                                                                      test_size=self.val_fraction,
                                                                                      random_state=42, shuffle=True)

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X_train,
                                                                                        self.__y_train,
                                                                                        test_size=test_fraction,
                                                                                        random_state=42, shuffle=True)

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

    @property
    def x_train(self):
        return self.__X_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def x_val(self):
        return self.__X_val

    @property
    def y_val(self):
        return self.__y_val

    @property
    def x_test(self):
        return self.__X_test

    @property
    def y_test(self):
        return self.__y_test


if __name__ == "__main__":
    dp = DataProcessor()
