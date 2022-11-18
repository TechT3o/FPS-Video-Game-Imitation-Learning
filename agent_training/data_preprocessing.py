from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Tuple
import numpy as np
import os


class DataProcessor:
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
    __y_train: np.ndarray
    __X_val: np.ndarray
    __y_val: np.ndarray
    __X_test: np.ndarray
    __y_test: np.ndarray

    def __init__(self, data_path: str, image_size: Tuple[int, int], color_channels: int, normalize: bool,
                 validation_fraction: float, test_fraction: float, time_steps=None):

        self.color_channels = color_channels
        self.__label_indices = dict()
        self.data_path = data_path if len(data_path) != 0 else os.getcwd()
        self.image_size = image_size
        self.time_steps = 0 if time_steps is None else time_steps
        self.val_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.normalize = False if normalize is None else normalize

    def load_dataset(self) -> np.ndarray:
        dataset = np.array([])
        return dataset

    def get_image(self, img_path) -> np.ndarray:
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode="rgb", target_size=self.image_size,
                                                    interpolation="lanczos")
        img = tf.keras.preprocessing.image.img_to_array(img, data_format=None)
        return img

    def reshape_dataset(self) -> None:
        if self.time_steps != 0 and len(self.__y.shape) == 1:
            self.__X = self.__X.reshape((-1, self.time_steps, self.image_size[0],
                                        self.image_size[1], self.color_channels))
            self.__y = self.__y.reshape(-1, self.time_steps)
            self.__y = self.__y = np.mean(self.__y, axis=1)
        elif self.time_steps != 0:
            self.__X = self.__X.reshape((-1, self.time_steps, self.image_size[0],
                                        self.image_size[1], self.color_channels))
            self.__y = self.__y.reshape((-1, self.time_steps, self.__y.shape[-1]))
            self.__y = self.__y = np.mean(self.__y, axis=1)

    def train_test_val_split(self) -> None:
        test_fraction = self.test_fraction / (1 - self.val_fraction)

        self.__X_train, self.__X_val, self.__y_train, self.__y_val = train_test_split(self.__X, self.__y,
                                                                                      test_size=self.val_fraction,
                                                                                      random_state=42, shuffle=True,
                                                                                      stratify=self.__y)

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X_train,
                                                                                        self.__y_train,
                                                                                        test_size=test_fraction,
                                                                                        random_state=42,
                                                                                        shuffle=True,
                                                                                        stratify=self.__y_train)

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
