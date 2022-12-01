import tensorflow as tf
from parameters import Parameters
from data_preprocessing import DataProcessor
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
        self.data_processor = DataProcessor()

        self.agent = tf.keras.models.load_model(os.path.join(self.agent_path, 'agent.h5'))
        print(self.agent.layers)

    def remove_feature_chain(self):
        self.agent = 0

    def fine_tune(self):
        self.agent.fit()


if __name__ == "__main__":
    learner = TransferLearner()
