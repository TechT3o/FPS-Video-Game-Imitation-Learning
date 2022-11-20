from tensorflow import keras
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, LSTM, Flatten, Input, TimeDistributed, concatenate, Dropout
from typing import Tuple


class ModelBuilder:
    """
    class that builds a model and evaluates it based on given flags
    """
    model_base: str
    lstm_flag: str
    base: str
    input_shape: Tuple[int, int, int, int]
    n_mouse_y: int
    n_mouse_x: int
    n_clicks: int
    n_features: int
    model: keras.layers.Layer

    def __init__(self, image_size: Tuple[int, int], time_steps: int, channel_number: int,
                 base: str = 'LeNet5', lstm_flag: str = 'LSTM'):
        """
        class constructor
        :param image_size: size of input images
        :param time_steps: length of image sequences given for training the model
        :param channel_number: size of channels that the input image has
        :param base: flag for which base to use available ones are 'EfficientNet' and 'LeNet5'
        :param lstm_flag: flag for model output to include an 'LSTM' layer
        """
        self.lstm_flag = lstm_flag
        self.base = base

        self.input_shape = (time_steps, image_size[0], image_size[1], channel_number)
        self.n_mouse_y = 12
        self.n_mouse_x = 19
        self.n_clicks = 2
        self.n_features = 1

        self.model = self._build_output_chains()

    def _build_base_model(self) -> Model:
        """
        Builds the CNN base model
        :return: keras model object
        """
        # TODO put other bases as well
        if self.base == 'EfficientNet':
            base_model = EfficientNetB0(weights='imagenet', input_shape=(self.input_shape[1:]), include_top=False,
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
        x = TimeDistributed(intermediate_model)(input_1)

        # x = ConvLSTM2D(filters=512, kernel_size=(3, 3), stateful=False, return_sequences=True,
        #                dropout=0.5, recurrent_dropout=0.5)(x)

        flat = TimeDistributed(Flatten())(x)

        x = LSTM(256, stateful=False, return_sequences=True, dropout=0., recurrent_dropout=0.)(flat)
        x = TimeDistributed(Dropout(0.5))(x)

        # 3) add shared fc layers
        dense_5 = x

        # 4) set up outputs, separate outputs will allow separate losses to be applied
        flat = Dense(32, activation='relu')(flat)
        output_1 = Dense(self.n_features, activation='sigmoid')(flat)
        output_2 = TimeDistributed(Dense(self.n_clicks, activation='sigmoid'))(dense_5)
        output_3 = TimeDistributed(Dense(self.n_mouse_x, activation='softmax'))(
            dense_5)  # softmax since mouse is mutually exclusive
        output_4 = TimeDistributed(Dense(self.n_mouse_y, activation='softmax'))(dense_5)
        # output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5)
        output_all = concatenate([output_2, output_3, output_4], axis=-1)
        # output_all = concatenate([output_1, output_2, output_3, output_4, output_5], axis=-1)

        # 5) finish model definition
        model = Model(input_1, [output_1, output_all])
        print(model.summary())
        return model


if __name__ == "__main__":
    builder = ModelBuilder((224, 121), 10, 3, base='EfficientNet', lstm_flag='')
