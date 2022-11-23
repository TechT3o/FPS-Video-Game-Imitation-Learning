from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Flatten, Input, TimeDistributed, concatenate, Dropout

from keras.optimizers import Adam
from keras.applications import EfficientNetB0, MobileNetV3Small
from keras.losses import CategoricalCrossentropy
from typing import Tuple


class ModelBuilder:
    """
    class that builds a model and evaluates it based on given flags
    """
    model_base: str
    lstm_flag: str
    base: str
    input_shape: Tuple[int, int, int, int] or Tuple[int, int, int]
    n_mouse_y: int
    n_mouse_x: int
    n_clicks: int
    n_features: int
    model: keras.layers.Layer

    def __init__(self, image_size: Tuple[int, int], time_steps: int, channel_number: int,
                 base: str = 'MobileNetv3', lstm_flag: str = 'LSTM', feature_chain_flag: bool = True):
        """
        class constructor
        :param image_size: size of input images
        :param time_steps: length of image sequences given for training the model
        :param channel_number: size of channels that the input image has
        :param base: flag for which base to use available ones are 'EfficientNet' and 'LeNet5'
        :param lstm_flag: flag for model output to include an 'LSTM' layer
        """
        self.lstm_flag = lstm_flag
        self.feature_chain_flag = feature_chain_flag
        self.base = base

        self.n_mouse_y = 13
        self.n_mouse_x = 18
        self.n_clicks = 2
        self.n_features = 1

        if self.lstm_flag == 'LSTM':
            self.input_shape = (time_steps, image_size[0], image_size[1], channel_number)
            self.model = self._build_recurrent_output_chains()
        else:
            self.input_shape = (image_size[0], image_size[1], channel_number)
            self.model = self._build_output_chains()

    def _build_base_model(self) -> Model:
        """
        Builds the CNN base model
        :return: keras model object
        """
        if self.base == 'MobileNetv3':

            if self.lstm_flag == 'LSTM':
                base_model = MobileNetV3Small(input_shape=self.input_shape[1:], alpha=1.0, minimalistic=False,
                                              include_top=False, weights="imagenet")
            else:
                base_model = MobileNetV3Small(input_shape=self.input_shape, alpha=1.0, minimalistic=False,
                                              include_top=False, weights="imagenet")
            base_model.trainable = True

            intermediate_model = Model(inputs=base_model.input, outputs=base_model.output)
            intermediate_model.trainable = True

        if self.base == 'EfficientNetB0':

            if self.lstm_flag == 'LSTM':
                base_model = EfficientNetB0(weights='imagenet', input_shape=self.input_shape[1:], include_top=False,
                                            drop_connect_rate=0.2)
            else:
                base_model = EfficientNetB0(weights='imagenet', input_shape=self.input_shape, include_top=False,
                                            drop_connect_rate=0.2)

            base_model.trainable = True

            intermediate_model = Model(inputs=base_model.input, outputs=base_model.layers[161].output)
            intermediate_model.trainable = True
        return intermediate_model

    def _build_output_chains(self) -> Model:
        """
        Builds the fully connected or recurrent output chains of the model
        :return: keras model object
        """
        intermediate_model = self._build_base_model()
        input_1 = Input(shape=self.input_shape, name='main_in')
        # x = TimeDistributed(intermediate_model)(input_1)

        # x = ConvLSTM2D(filters=512, kernel_size=(3, 3), stateful=False, return_sequences=True,
        #                dropout=0.5, recurrent_dropout=0.5)(x)

        flat = Flatten()(input_1)

        # 3) add shared fc layers
        dense_5 = flat

        # 4) set up outputs, separate outputs will allow separate losses to be applied

        if self.feature_chain_flag:
            flat = Dense(32, activation='relu')(flat)
            output_1 = Dense(self.n_features, activation='sigmoid')(flat)

        output_2 = Dense(self.n_clicks, activation='sigmoid', name='click_out')(dense_5)
        output_3 = Dense(self.n_mouse_x, activation='softmax', name='mouse_x_out')(dense_5)  # softmax since mouse is mutually exclusive
        output_4 = Dense(self.n_mouse_y, activation='softmax', name='mouse_y_out')(dense_5)
        # output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5)
        output_all = [output_2, output_3, output_4]
        # output_all = concatenate([output_2, output_3, output_4], axis=-1)
        # output_all = concatenate([output_1, output_2, output_3, output_4, output_5], axis=-1)

        # 5) finish model definition
        if self.feature_chain_flag:
            model = Model(input_1, [output_1, output_all])
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy(), 'features_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy", 'features_out': "accuracy"}
        else:
            model = Model(input_1, output_all)
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy"}

        model.compile(optimizer=Adam(1e-3), loss=loss, metrics=metrics)

        print(model.summary())
        return model

    def _build_recurrent_output_chains(self) -> Model:
        """
        Builds the fully connected or recurrent output chains of the model
        :return: keras model object
        """
        intermediate_model = self._build_base_model()
        input_1 = Input(shape=self.input_shape, name='main_in')
        x = TimeDistributed(intermediate_model)(input_1)

        # x = ConvLSTM2D(filters=512, kernel_size=(3, 3), stateful=False, return_sequences=True,
        #                dropout=0.5, recurrent_dropout=0.5)(x)

        flat = TimeDistributed(Flatten())(x)

        x = LSTM(256, stateful=False, return_sequences=True, dropout=0., recurrent_dropout=0.)(flat)
        x = TimeDistributed(Dropout(0.5))(x)

        # 3) add shared fc layers
        dense_5 = x

        # 4) set up outputs, separate outputs will allow separate losses to be applied

        if self.feature_chain_flag:
            flat = Dense(32, activation='relu')(flat)
            output_1 = Dense(self.n_features, activation='sigmoid', name='features_out')(flat)

        output_2 = TimeDistributed(Dense(self.n_clicks, activation='sigmoid'), name='click_out')(dense_5)
        output_3 = TimeDistributed(Dense(self.n_mouse_x, activation='softmax'), name='mouse_x_out')(dense_5)
        # softmax since mouse is mutually exclusive
        output_4 = TimeDistributed(Dense(self.n_mouse_y, activation='softmax'), name='mouse_y_out')(dense_5)
        # output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5)
        output_all = [output_2, output_3, output_4]
        # output_all = concatenate([output_2, output_3, output_4], axis=-1)
        # output_all = concatenate([output_1, output_2, output_3, output_4, output_5], axis=-1)

        # 5) finish model definition
        if self.feature_chain_flag:
            model = Model(input_1, [output_1, output_all])
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy(), 'features_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy", 'features_out': "accuracy"}
        else:
            model = Model(input_1, output_all)
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy"}

        model.compile(optimizer=Adam(1e-3), loss=loss, metrics=metrics)
        print(model.summary())
        return model


if __name__ == "__main__":
    builder = ModelBuilder((224, 121), 10, 3, base='EfficientNetB0', lstm_flag='')
