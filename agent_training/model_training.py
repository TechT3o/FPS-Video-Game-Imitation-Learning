from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from agent_training.model_building import ModelBuilder
from agent_training.data_preprocessing import DataProcessor
from agent_training.data_generator import DataGenerator
from typing import Tuple
import tensorflow as tf
from agent_training.parameters import Parameters
import numpy as np


class ModelTrainer:
    model_builder: ModelBuilder
    dataset: DataProcessor
    __model: tf.keras.Model
    augmentation: bool
    save_path: str
    data_path: str
    BATCH_SIZE: int
    __metrics: dict
    __history: dict

    def __init__(self):
        """
        class constructor
        :param save_path: path to save the model built
        :param model_base: flag for which model base to use
        :param augmentation: flag to augment input data or not
        """

        self.params = Parameters()
        self.save_path = self.params.save_path
        self.data_path = self.params.data_path
        self.color_channels = self.params.channel_size
        self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction
        self.test_fraction = self.params.test_fraction
        self.lstm_flag = self.params.lstm_flag
        self.feature_chain_flag = self.params.feature_chain_flag
        self.base = self.params.model_base
        self.augmentation = self.params.augmentation
        self.loading_flag = self.params.loading_flag
        self.BATCH_SIZE = self.params.batch_size

        if self.lstm_flag == 'LSTM' or self.time_steps > 0:
            assert self.lstm_flag == 'LSTM' and self.time_steps > 0

        if self.loading_flag == 'generator':
            self.train_generator = DataGenerator(data_flag='train')
            self.validation_generator = DataGenerator(data_flag='validation')
        self.dataset = DataProcessor()
        self.model_builder = ModelBuilder(self.dataset.mouse_x_len, self.dataset.mouse_y_len, self.dataset.clicks_len)
        self.__model = self.model_builder.model
        # self.BATCH_SIZE = self.dataset.x_val.shape[0] // 100 if self.dataset.x_val.shape[0] >= 100 else \
        #     self.dataset.x_val.shape[0]
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
        # if self.dataset.x_train.max() > 1:
        #     x_train *= 1. / 255.
        #     x_val *= 1. / 255.

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
            # print(self.dataset.y_train.shape, self.dataset.y_val.shape)
            self.__history = self.__model.fit(x=x_train, y=self.dataset.y_train, batch_size=self.BATCH_SIZE,
                                              epochs=50000, callbacks=callbacks,
                                              validation_data=(x_val, self.dataset.y_val),
                                              steps_per_epoch=self.dataset.x_train.shape[
                                                                  0] // self.BATCH_SIZE,
                                              validation_steps=self.dataset.x_val.shape[
                                                                   0] // self.BATCH_SIZE,
                                              verbose=1)
            self.__model = tf.keras.models.load_model(self.save_path + "\\model.h5")

    def _train_model_gen(self) -> None:
        """
        Trains the model for specific callbacks and saves the training history and the best model
        :return: None
        """

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(self.save_path + "\\model.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger(self.save_path + '\\training.log'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=40, verbose=1,
                                             mode="min")]

        self.__history = self.__model.fit_generator(generator=self.train_generator,
                                                    validation_data=self.validation_generator, epochs=50000,
                                                    steps_per_epoch=self.dataset.x_train.shape[0] // self.BATCH_SIZE,
                                                    validation_steps=self.dataset.x_val.shape[0] // self.BATCH_SIZE,
                                                    callbacks=callbacks, workers=6)

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
        click_test, mouse_x_test, mouse_y_test = self.dataset.y_test
        click_pred, mouse_x_pred, mouse_y_pred = self._predict(x_test)

        x_predictions = np.round(mouse_x_pred)
        y_predictions = np.round(mouse_y_pred)
        click_predictions = np.round(click_pred)

        mouse_y_test = mouse_y_test.reshape((mouse_y_test.shape[0] * mouse_y_test.shape[1], mouse_y_test.shape[2]))
        mouse_x_test = mouse_x_test.reshape((mouse_x_test.shape[0] * mouse_x_test.shape[1], mouse_x_test.shape[2]))
        click_test = click_test.reshape((click_test.shape[0] * click_test.shape[1], click_test.shape[2]))

        click_predictions = click_predictions.reshape((click_predictions.shape[0] * click_predictions.shape[1], click_predictions.shape[2]))
        x_predictions = x_predictions.reshape((x_predictions.shape[0] * x_predictions.shape[1], x_predictions.shape[2]))
        y_predictions = y_predictions.reshape((y_predictions.shape[0] * y_predictions.shape[1], y_predictions.shape[2]))

        print(mouse_x_test.shape, x_predictions.shape)
        print(mouse_y_test.shape, y_predictions.shape)
        print(click_test.shape, click_predictions.shape)

        self.__metrics["mouse_x_f1_score"] = f1_score(mouse_x_test, x_predictions, average=None)
        # self.__metrics["mouse_x_precision"] = precision_score(mouse_x_test, x_predictions, average=None)
        self.__metrics["mouse_x_recall"] = recall_score(mouse_x_test, x_predictions, average=None)
        # self.__metrics["mouse_x_roc_auc"] = roc_auc_score(mouse_x_test, x_predictions, average=None)
        self.__metrics["mouse_x_accuracy"] = accuracy_score(mouse_x_test, x_predictions)

        self.__metrics["mouse_y_f1_score"] = f1_score(mouse_y_test, y_predictions, average=None)
        # self.__metrics["mouse_y_precision"] = precision_score(mouse_y_test, y_predictions, average=None)
        self.__metrics["mouse_y_recall"] = recall_score(mouse_y_test, y_predictions, average=None)
        # self.__metrics["mouse_y_roc_auc"] = roc_auc_score(mouse_y_test, y_predictions, average=None)
        self.__metrics["mouse_y_accuracy"] = accuracy_score(mouse_y_test, y_predictions)

        self.__metrics["click_f1_score"] = f1_score(click_test, click_predictions, average=None)
        # self.__metrics["click_precision"] = precision_score(click_test, click_predictions, average=None)
        self.__metrics["click_recall"] = recall_score(click_test, click_predictions, average=None)
        # self.__metrics["click_roc_auc"] = roc_auc_score(click_test, click_predictions, average=None)
        self.__metrics["click_accuracy"] = accuracy_score(click_test, click_predictions)

        print(self.__metrics)

    def _train_and_evaluate_model(self) -> None:
        """
        Trains and evaluates the model
        :return: None
        """
        if self.loading_flag == 'generator':
            self._train_model_gen()
        else:
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
