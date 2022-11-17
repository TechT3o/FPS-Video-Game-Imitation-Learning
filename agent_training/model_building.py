from tensorflow import keras
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, LSTM, Flatten, Conv2D, InputLayer, Input, TimeDistributed, ConvLSTM2D


class ModelBuilder:
    def __init__(self, base='LeNet5', lstm_flag='LSTM'):
        self.model_base = None
        self.lstm_flag = lstm_flag
        self.base = base

    def efficientnet_base(self):
        self.model_base = EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(258, 128, 3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax")

    def lenet5_base(self):

        input = Input(shape=(252, 121, 3))
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=2, input_shape=(252, 121, 3), activation='relu', name='conv_1')(input)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=2, activation='relu', name='conv_2')(conv1)
        flattened = Flatten(name= 'flatten')(conv2)
        chain_1 = TimeDistributed(flattened, activation="relu", name="layer1")
        lstm = ConvLSTM2D(256, stateful=False, return_sequences=True, dropout=0., recurrent_dropout=0.)(chain_1)
        chain_2 = Dense(256, activation="relu", name="layer2")(flattened)
        out_1 = Dense(1, name="out1")(lstm)
        out_2 = Dense(3, name="out2")(chain_2)

        model = Model(inputs=input, outputs=[out_1, out_2])
        print(model.summary())

    def lenet5_base_seq(self):

        model = keras.Sequential([
        Conv2D(filters=16, strides=2, kernel_size=(3, 3), input_shape=(252, 121, 3), activation='relu', name='conv_1'),
        Conv2D(filters=64,strides=2, kernel_size=(3, 3), activation='relu', name='conv_2'),
        Flatten(name='flatten'),
        Dense(256, activation="relu", name="layer1"),
        Dense(3, name="out2")

        ])
        model.summary()

    def game_feature_chain(self):
        self.x = self.model_base.output
        self.x = Flatten()(self.x)
        self.x = Dense(256, activation='relu')(self.x)
        self.x = Dense(1, activation='sigmoid')(self.x)

    def action_scores_chain(self):
        self.y = self.model_base.output
        self.y = Flatten()(self.y)
        self.y = Dense(1024, activation='relu')(self.y)

        if self.lstm_flag == 'LSTM':
            self.y = LSTM(4)(self.y)

        self.y = Dense(3, activation='sigmoid')(self.y)

    def build_model(self):
        if self.base == 'EfficientNet':
            self.efficientnet_base()
        if self.base == 'LeNet5':
            self.lenet5_base()
        self.game_feature_chain()
        self.action_scores_chain()
        model = Model(inputs=self.model_base.input, outputs=[self.x, self.y])
        model.summary()

builder = ModelBuilder()
#builder.build_model()
builder.lenet5_base()

