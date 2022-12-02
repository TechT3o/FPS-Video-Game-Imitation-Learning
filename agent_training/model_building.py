from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, LSTM, Flatten, Input, TimeDistributed, concatenate, Dropout, Conv2D, BatchNormalization
from agent_training.parameters import Parameters
from keras.optimizers import Adam
from keras.applications import EfficientNetB0, MobileNetV3Small, ResNet50V2, NASNetMobile
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
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

    def __init__(self, mouse_x: int, mouse_y: int, clicks: int, features: int = 0):
        """
        class constructor
        :param image_size: size of input images
        :param time_steps: length of image sequences given for training the model
        :param channel_number: size of channels that the input image has
        :param base: flag for which base to use available ones are 'EfficientNet' and 'LeNet5'
        :param lstm_flag: flag for model output to include an 'LSTM' layer
        """

        self.params = Parameters()
        self.data_path = self.params.data_path
        self.color_channels = self.params.channel_size
        self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction
        self.test_fraction = self.params.test_fraction
        self.lstm_flag = self.params.lstm_flag
        self.feature_chain_flag = self.params.feature_chain_flag
        self.base = self.params.model_base

        self.n_mouse_y = mouse_y
        self.n_mouse_x = mouse_x
        self.n_clicks = clicks
        self.n_features = features

        if self.lstm_flag == 'LSTM':
            self.input_shape = (self.time_steps, self.image_size[0], self.image_size[1], self.color_channels)
            self.model = self._build_recurrent_output_chains()
        else:
            self.input_shape = (self.image_size[0], self.image_size[1], self.color_channels)
            self.model = self._build_output_chains()

    def _build_base_model(self) -> Model:
        """
        Builds the CNN base model
        :return: keras model object
        """

        if self.base == "NASNetMobile":
            print("NASNetMobile")
            if self.lstm_flag == 'LSTM':
                base_model = NASNetMobile(include_top=False, weights="imagenet",
                                        input_shape=self.input_shape[1:], pooling=None)
            else:
                base_model = NASNetMobile(include_top=False, weights="imagenet",
                                        input_shape=self.input_shape, pooling=None)
            base_model.trainable = True
            intermediate_model = Model(inputs=base_model.input, outputs=base_model.output)
            intermediate_model.trainable = True
        elif self.base == 'ResNet50V2':
            print('ResNet50v2')

            if self.lstm_flag == 'LSTM':
                base_model = ResNet50V2(include_top=False, weights="imagenet",
                                        input_shape=self.input_shape[1:], pooling=None)
            else:
                base_model = ResNet50V2(include_top=False, weights="imagenet",
                                        input_shape=self.input_shape, pooling=None)
            base_model.trainable = True

            intermediate_model = Model(inputs=base_model.input, outputs=base_model.output)
            intermediate_model.trainable = True
        elif self.base == 'MobileNetv3':
            print('MobileNetv3')

            if self.lstm_flag == 'LSTM':
                base_model = MobileNetV3Small(input_shape=self.input_shape[1:], alpha=1.0, minimalistic=False,
                                              include_top=False, weights="imagenet")
            else:
                base_model = MobileNetV3Small(input_shape=self.input_shape, alpha=1.0, minimalistic=False,
                                              include_top=False, weights="imagenet")
            base_model.trainable = True

            intermediate_model = Model(inputs=base_model.input, outputs=base_model.output)
            intermediate_model.trainable = True

        elif self.base == 'EfficientNetB0':
            print('EfficientNetB0')
            if self.lstm_flag == 'LSTM':
                base_model = EfficientNetB0(weights='imagenet', input_shape=self.input_shape[1:], include_top=False,
                                            drop_connect_rate=0.2)
            else:
                base_model = EfficientNetB0(weights='imagenet', input_shape=self.input_shape, include_top=False,
                                            drop_connect_rate=0.2)

            base_model.trainable = True

            intermediate_model = Model(inputs=base_model.input, outputs=base_model.layers[161].output)
            intermediate_model.trainable = True
        else:
            print('CNN')
            if self.lstm_flag == 'LSTM':
                input_conv = Input(shape=self.input_shape[1:])
            else:
                input_conv = Input(shape=self.input_shape)

            conv = Conv2D(12, (5, 5), padding='same', strides=2, activation='relu')(input_conv)
            # MaxPooling2D((2, 2), strides=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.4)(conv)
            conv = Conv2D(12, (3, 3), strides=2, padding='same', activation='relu')(conv)
            # MaxPooling2D((2, 2), strides=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.4)(conv)
            conv = Conv2D(12*2, (3, 3), strides=2, padding='same', activation='relu')(conv)
            # MaxPooling2D((2, 2), strides=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.4)(conv)
            conv = Conv2D(12*2, (3, 3), strides=2, padding='same', activation='relu')(conv)
            # MaxPooling2D((2, 2), strides=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.4)(conv)
            conv = Conv2D(12*3, (3, 3), strides=2, padding='same', activation='relu')(conv)
            # MaxPooling2D((2, 2), strides=2)(conv)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.4)(conv)
            conv = Flatten()(conv)
            intermediate_model = Model(inputs=input_conv, outputs=conv)
        return intermediate_model

    def _build_output_chains(self) -> Model:
        """
        Builds the fully connected or recurrent output chains of the model
        :return: keras model object
        """
        intermediate_model = self._build_base_model()
        input_1 = Input(shape=self.input_shape, name='main_in')
        x = intermediate_model(input_1)

        # x = ConvLSTM2D(filters=512, kernel_size=(3, 3), stateful=False, return_sequences=True,
        #                dropout=0.5, recurrent_dropout=0.5)(x)

        flat = Flatten()(input_1)

        # 3) add shared fc layers
        dense_5 = flat

        # 4) set up outputs, separate outputs will allow separate losses to be applied

        output_2 = Dense(self.n_clicks, activation='sigmoid', name='click_out')(dense_5)
        output_3 = Dense(self.n_mouse_x, activation='softmax', name='mouse_x_out')(dense_5)  # softmax since mouse is mutually exclusive
        output_4 = Dense(self.n_mouse_y, activation='softmax', name='mouse_y_out')(dense_5)
        # output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5)
        output_all = [output_2, output_3, output_4]
        # output_all = concatenate([output_2, output_3, output_4], axis=-1)
        # output_all = concatenate([output_1, output_2, output_3, output_4, output_5], axis=-1)
        if self.feature_chain_flag:
            output_1 = Dense(self.n_features, activation='softmax')(flat)
            output_all = [output_1, output_2, output_3, output_4]
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy(), 'features_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy", 'features_out': "accuracy"}
        else:
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy"}
        model = Model(input_1, output_all)
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
        x = TimeDistributed(intermediate_model, name='base_model')(input_1)

        # x = ConvLSTM2D(filters=512, kernel_size=(3, 3), stateful=False, return_sequences=True,
        #                dropout=0.5, recurrent_dropout=0.5)(x)

        flat = TimeDistributed(Flatten(), name='flatten')(x)

        x = LSTM(256, stateful=False, return_sequences=True, dropout=0., recurrent_dropout=0., name='lstm')(flat)
        x = TimeDistributed(Dropout(0.5))(x)

        # 3) add shared fc layers
        dense_5 = x

        # 4) set up outputs, separate outputs will allow separate losses to be applied

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
            output_1 = TimeDistributed(Dense(self.n_features, activation='softmax'), name='features_out')(flat)
            output_all = [output_1, output_2, output_3, output_4]
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy(), 'features_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy", 'features_out': "accuracy"}
        else:
            loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                    'click_out': CategoricalCrossentropy()}
            metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                       'click_out': "accuracy"}
        model = Model(input_1, output_all)
        model.compile(optimizer=Adam(1e-3), loss=loss, metrics=metrics)
        print(model.summary())
        return model


if __name__ == "__main__":
    builder = ModelBuilder(18, 13, 2, 1)
