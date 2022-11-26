# code adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
from tensorflow import keras
from agent_training.data_preprocessing import DataProcessor
from agent_training.parameters import Parameters


class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=32, n_classes=10, shuffle=True):

        self.parameters = Parameters()
        self.data_path = self.params.data_path
        # self.image_size = (self.params.image_size_x, self.params.image_size_y)
        self.time_steps = self.params.time_steps
        self.val_fraction = self.params.validation_fraction
        self.test_fraction = self.params.test_fraction

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.color_channels = self.params.channel_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_processor = DataProcessor()
        self.list_IDs = self.data_processor.image_paths

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.color_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)