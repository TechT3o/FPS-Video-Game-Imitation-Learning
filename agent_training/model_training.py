from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from model_building import ModelBuilder
from data_preprocessing import DataProcessor
import tensorflow as tf
import os
import numpy as np


class ModelTrainer:
    model_builder: ModelBuilder
    dataset: DataProcessor
    __model: tf.keras.Model
    augmentation: bool
    save_path: str
    BATCH_SIZE: int
    __metrics: dict
    __history: dict

    def __init__(self, save_path: str, model_base: str = 'EfficientNet', augmentation: bool = None):
        """
        class constructor
        :param save_path: path to save the model built
        :param model_base: flag for which model base to use
        :param augmentation: flag to augment input data or not
        """
        self.model_builder = ModelBuilder(image_size=(252, 121), time_steps=10, channel_number=3,
                                          base=model_base, lstm_flag='')
        self.dataset = DataProcessor(data_path='', color_channels=1, image_size=(252, 121), normalize=True,
                                     time_steps=10, test_fraction=0.2, validation_fraction=0.2)
        self.__model = self.model_builder.model
        self.augmentation = False if augmentation is None else augmentation
        self.save_path = os.getcwd() if save_path is None else save_path
        self.BATCH_SIZE = self.dataset.x_val.shape[0] // 100 if self.dataset.x_val.shape[0] >= 100 else \
            self.dataset.x_val.shape[0]
        self.__metrics = {}
        self._train_and_evaluate_model()

    def image_generator(self) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Keras generator that takes the datasetX augments it and the inputs it in the model training
        :return: keras image data generator object
        """
        return tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=(-0.1, 0.1),
            shear_range=0.0,
            zoom_range=0.0,
            channel_shift_range=0.0,
            fill_mode='nearest',
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0,
            interpolation_order=1,
            dtype=None
        )

    def _train_model(self) -> None:
        """
        Trains the model for specific callbacks and saves the training history and the best model
        :return: None
        """
        x_train = self.dataset.x_train
        x_val = self.dataset.x_val
        if self.dataset.x_train.max() > 1:
            x_train *= 1. / 255.
            x_val *= 1. / 255.

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(self.save_path + "\\model.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger(self.save_path + '\\training.log'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=40, verbose=1,
                                             mode="min")]

        if self.augmentation:
            data_generator = self.image_generator()
            self.__history = self.__model.fit(x=data_generator.flow(self.dataset.x_train, self.dataset.y_train,
                                                                    batch_size=self.BATCH_SIZE),
                                              epochs=50000, callbacks=callbacks,
                                              validation_data=data_generator.flow(self.dataset.x_val,
                                                                                  self.dataset.y_val,
                                                                                  batch_size=self.BATCH_SIZE),
                                              steps_per_epoch=self.dataset.x_train.shape[
                                                                  0] // self.BATCH_SIZE,
                                              validation_steps=self.dataset.x_val.shape[
                                                                   0] // self.BATCH_SIZE,
                                              verbose=1)
            self.__model = tf.keras.models.load_model(self.save_path + "\\model.h5")
        else:
            self.__history = self.__model.fit(x=x_train, y=self.dataset.y_train, batch_size=self.BATCH_SIZE,
                                              epochs=50000, callbacks=callbacks,
                                              validation_data=(x_val, self.dataset.y_val),
                                              steps_per_epoch=self.dataset.x_train.shape[
                                                                  0] // self.BATCH_SIZE,
                                              validation_steps=self.dataset.x_val.shape[
                                                                   0] // self.BATCH_SIZE,
                                              verbose=1)
            self.__model = tf.keras.models.load_model(self.save_path + "\\model.h5")

    def _predict(self, images: np.ndarray) -> np.ndarray:
        """
        Uses the model to make a prediction for the input image
        :param images: input image to test
        :return: prediction in an array
        """
        return self.__model.predict(images)

    def _evaluate(self) -> None:
        """
        Tests the trained model on the test dataset and retrieves models metrics
        :return: None
        """
        x_test = self.dataset.x_test
        y_test = self.dataset.y_test
        predictions = self._predict(x_test)
        if len(y_test.shape) == 1:
            predictions = predictions[:, 0]
        predictions = np.round(predictions)
        self.__metrics["f1_score"] = f1_score(y_test, predictions, average=None)
        self.__metrics["precision"] = precision_score(y_test, predictions, average=None)
        self.__metrics["recall"] = recall_score(y_test, predictions, average=None)
        self.__metrics["roc_auc"] = roc_auc_score(y_test, predictions, average=None)
        self.__metrics["accuracy"] = accuracy_score(y_test, predictions)

    def _train_and_evaluate_model(self) -> None:
        """
        Trains and evaluates the model
        :return: None
        """
        self._train_model()
        self._evaluate()

    @property
    def history(self):
        return self.__history

    @property
    def metrics(self):
        return self.__metrics

    @property
    def model(self):
        return self.__model
