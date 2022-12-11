from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from agent_training.model_building import ModelBuilder
from agent_training.data_preprocessing import DataProcessor
from agent_training.data_generator import DataGenerator
import tensorflow as tf
from agent_training.parameters import Parameters
import numpy as np


class ModelTrainer:
    """
    Trains and evaluates models
    """
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
        self.loading_flag = self.params.loading_flag
        self.BATCH_SIZE = self.params.batch_size

        if self.lstm_flag == 'LSTM' or self.time_steps > 0:
            assert self.lstm_flag == 'LSTM' and self.time_steps > 0

        if self.loading_flag == 'generator':
            self.train_generator = DataGenerator(data_flag='training')
            self.validation_generator = DataGenerator(data_flag='validation')
            self.model_builder = ModelBuilder(self.train_generator.mouse_x_len, self.train_generator.mouse_y_len,
                                              self.train_generator.clicks_len, self.train_generator.features_len)
            print(self.train_generator.features_len, self.validation_generator.features_len)
        else:
            self.dataset = DataProcessor()
            self.model_builder = ModelBuilder(self.dataset.mouse_x_len, self.dataset.mouse_y_len,
                                              self.dataset.clicks_len, self.dataset.features_len)

        self.__model = self.model_builder.model
        self.__metrics = {}
        self._train_and_evaluate_model()

    def _train_model(self) -> None:
        """
        Trains the model for specific callbacks and saves the training history and the best model
        :return: None
        """
        x_train = self.dataset.x_train
        x_val = self.dataset.x_val

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(self.save_path + "\\model.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger(self.save_path + '\\training.log'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=40, verbose=1,
                                             mode="min")]

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
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=20, verbose=1,
                                             mode="min")]

        self.__history = self.__model.fit_generator(generator=self.train_generator,
                                                    validation_data=self.validation_generator, epochs=50000,
                                                    steps_per_epoch=self.train_generator.data_size // self.BATCH_SIZE,
                                                    validation_steps=self.validation_generator.data_size // self.BATCH_SIZE,
                                                    callbacks=callbacks, workers=8)

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

        click_predictions = click_predictions.reshape((click_predictions.shape[0] * click_predictions.shape[1],
                                                       click_predictions.shape[2]))
        x_predictions = x_predictions.reshape((x_predictions.shape[0] * x_predictions.shape[1], x_predictions.shape[2]))
        y_predictions = y_predictions.reshape((y_predictions.shape[0] * y_predictions.shape[1], y_predictions.shape[2]))

        self.__metrics["mouse_x_f1_score"] = f1_score(mouse_x_test, x_predictions, average=None)
        self.__metrics["mouse_x_recall"] = recall_score(mouse_x_test, x_predictions, average=None)
        self.__metrics["mouse_x_accuracy"] = accuracy_score(mouse_x_test, x_predictions)

        self.__metrics["mouse_y_f1_score"] = f1_score(mouse_y_test, y_predictions, average=None)
        self.__metrics["mouse_y_recall"] = recall_score(mouse_y_test, y_predictions, average=None)
        self.__metrics["mouse_y_accuracy"] = accuracy_score(mouse_y_test, y_predictions)

        self.__metrics["click_f1_score"] = f1_score(click_test, click_predictions, average=None)
        self.__metrics["click_recall"] = recall_score(click_test, click_predictions, average=None)
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
