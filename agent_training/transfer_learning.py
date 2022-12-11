import tensorflow as tf
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from agent_training.parameters import Parameters
from agent_training.data_preprocessing import DataProcessor
import os


class TransferLearner:
    def __init__(self):

        self.params = Parameters()
        self.agent_path = self.params.agent_path
        self.data_path = self.params.one_shot_path
        self.color_channels = self.params.channel_size
        self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction
        self.lstm_flag = self.params.lstm_flag
        self.feature_chain_flag = self.params.feature_chain_flag
        self.augmentation = self.params.augmentation
        self.BATCH_SIZE = self.params.batch_size

        self.agent = tf.keras.models.load_model(os.path.join(self.agent_path, 'agent.h5'))

        self.remove_feature_chain()
        print(self.agent.summary())

    def remove_feature_chain(self) -> None:
        """
        Finds if feature chain exists and rebuilds the model without it
        :return: None
        """
        for layer in self.agent.layers:
            if layer.name == 'features_out':
                intermediate_model = Model(self.agent.input, self.agent.layers[4].output)
                outputs = [self.agent.layers[6].output, self.agent.layers[7].output, self.agent.layers[8].output]
                self.agent = Model(intermediate_model.input, outputs)

    def compile_agent(self) -> None:
        """
        Compiles the model
        :return: None
        """
        loss = {'mouse_x_out': CategoricalCrossentropy(), 'mouse_y_out': CategoricalCrossentropy(),
                'click_out': BinaryCrossentropy()}
        metrics = {'mouse_x_out': "accuracy", 'mouse_y_out': "accuracy",
                   'click_out': "accuracy"}
        self.agent.compile(optimizer=Adam(1e-5), loss=loss, metrics=metrics)

    def fine_tune(self) -> None:
        """
        Fine-tunes the model with the new data
        :return:
        """
        print("Loading data")
        dataset = DataProcessor(self.data_path, transfer_flag=True)
        batch_size = 4

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("agent.h5", save_best_only=True, verbose=1),
            tf.keras.callbacks.CSVLogger('training.log'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=20, verbose=1,
                                             mode="min")]
        print("Compiling model")
        self.compile_agent()
        self.agent.fit(x=dataset.x_train, y=dataset.y_train, batch_size=batch_size, epochs=50000, callbacks=callbacks,
                                              validation_data=(dataset.x_val, dataset.y_val),
                                              steps_per_epoch=dataset.x_train.shape[0] // batch_size,
                                              validation_steps=dataset.x_val.shape[0] // batch_size,
                                              verbose=1)
        return


if __name__ == "__main__":
    learner = TransferLearner()
    learner.fine_tune()
