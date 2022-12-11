import os
import numpy as np
import pandas as pd
from singleton import Singleton
from statics import visualize_labels, add_in_between_elements
import matplotlib.pyplot as plt


class DataNormalizer(metaclass=Singleton):
    """
    class that reads the csv data gets the image paths and cleans and discretizes them to a form used to be classified
    """
    def __init__(self, data_path: str = '', game_feature_chain: int = 0, adjacent_labels: int = 0, frame_skip: int = 1,
                 transfer_flag: bool = False):
        """
        class constructor
        :param data_path: path where the csv files are stored
        """
        self.game_features_flag = game_feature_chain
        self.transfer_flag = transfer_flag
        self.adjacent_labels_encoding = adjacent_labels
        self.frame_skip = frame_skip
        self.csv_path = os.path.join(data_path, os.path.join('data', 'csvs'))
        self.action_space_x = np.array([-300, -200, -150, -100, -50, -25, -10,
                                        -5, -1, 0, 1, 5, 10, 50, 100, 150, 200, 300])
        self.action_space_y = np.array([-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100])

        self.one_hot_matrix_x = np.eye(len(self.action_space_x), len(self.action_space_x))
        self.one_hot_matrix_y = np.eye(len(self.action_space_y), len(self.action_space_y))

        self.discretize_x_function = np.vectorize(lambda x:
                                                  self.action_space_x[(np.abs(self.action_space_x - x)).argmin()])
        self.discretize_y_function = np.vectorize(lambda y:
                                                  self.action_space_y[(np.abs(self.action_space_y - y)).argmin()])
        self.data_dataframe = pd.DataFrame()
        self.load_csvs()

        # print(self.data_dataframe)
        self.keep_non_edge_data()
        self.data_dataframe['Delta X'].plot.hist(bins=19)
        plt.show()
        # self.data_dataframe['Delta Y'].plot.hist(bins=11)
        # print(self.data_dataframe['Delta X'].shape, self.data_dataframe['Delta X'])
        if self.frame_skip > 1:
            self.perform_frame_skip()
        # print(self.data_dataframe['Delta X'].shape, self.data_dataframe['Delta X'])
        self.data_dataframe['Delta X'] = self.discretize_x_function(self.data_dataframe['Delta X'].values)
        self.data_dataframe['Delta Y'] = self.discretize_y_function(self.data_dataframe['Delta Y'].values)
        # self.one_hot_encoding()

    def one_hot_encoding(self):
        """
        Encodes the available actions in one hot form to be used as classification labels
        :return: numpy arrays with the one-hot encoding for the x_motion, y_motion, click actions
        """
        if self.transfer_flag:
            x = self.data_dataframe['Delta X'].to_numpy()
            y = self.data_dataframe['Delta Y'].to_numpy()
            one_hot_x = self.one_hot_encode_x(x)
            one_hot_y = self.one_hot_encode_y(y)
        else:
            one_hot_x = pd.get_dummies(self.data_dataframe['Delta X']).to_numpy()
            one_hot_y = pd.get_dummies(self.data_dataframe['Delta Y']).to_numpy()
        one_hot_click = np.expand_dims(self.data_dataframe['Shot'].to_numpy(), axis=1)

        if self.adjacent_labels_encoding:
            one_hot_x = self.encode_adjacent_values(one_hot_x)
            one_hot_y = self.encode_adjacent_values(one_hot_y)

        visualize_labels(one_hot_x, self.action_space_x, title= 'X Motions Histogram')
        visualize_labels(one_hot_y, self.action_space_y, title= 'Y Motions Histogram')
        print(np.sum(one_hot_click)/len(one_hot_click))
        # visualize_labels(one_hot_click, ['No shoot', 'shoot'], title= 'Shooting Histogram')

        if self.game_features_flag:
            one_hot_features = pd.get_dummies(self.data_dataframe['Target no']).to_numpy()
            return one_hot_x, one_hot_y, one_hot_click, one_hot_features

        return one_hot_x, one_hot_y, one_hot_click

    def encode_adjacent_values(self, array):
        for index, element in enumerate(array):
            one_index = np.argmax(element)
            if one_index == 0:
                element[one_index+1] = 1
                array[index] = element
            elif one_index == len(element) -1:
                element[one_index - 1] = 1
                array[index] = element
            else:
                element[one_index - 1] = 1
                element[one_index + 1] = 1
                array[index] = element
        return array

    def load_csvs(self):
        """
        Loads the csv files in a pandas dataframe
        :return: None
        """
        if self.game_features_flag:
            self.data_dataframe = pd.read_csv(os.path.join(self.csv_path, "csv_w_targets.csv"))
        else:
            csvs = os.listdir(self.csv_path)[:1]
            try:
                csvs.remove("csv_w_targets.csv")
            except Exception as e:
                print(e)

            for csv_file in csvs:
                sample_dataframe = pd.read_csv(os.path.join(self.csv_path, csv_file))
                self.data_dataframe = pd.concat([self.data_dataframe, sample_dataframe], axis=0, ignore_index=True)

    def one_hot_encode_x(self, array):
        one_hot = []
        for element in array:
            one_hot.append(self.one_hot_matrix_x[np.where(self.action_space_x == element)[0]][0])
        return np.array(one_hot)

    def one_hot_encode_y(self, array):
        one_hot = []
        for element in array:
            one_hot.append(self.one_hot_matrix_y[np.where(self.action_space_y == element)[0]][0])
        return np.array(one_hot)

    def keep_non_edge_data(self):
        """
        Keeps only the data where the window edge was not hit
        :return: None
        """
        self.data_dataframe = self.data_dataframe[self.data_dataframe['Hit Edge Flag'] == False]

    def perform_frame_skip(self):

        skipped_dataframe = pd.DataFrame()
        skipped_dataframe['Delta X'] = add_in_between_elements(self.data_dataframe['Delta X'], self.frame_skip)
        skipped_dataframe['Delta Y'] = add_in_between_elements(self.data_dataframe['Delta Y'], self.frame_skip)
        skipped_dataframe['Image Path'] = self.data_dataframe['Image Path'].to_numpy().copy()[1::self.frame_skip]
        skipped_dataframe['Shot'] = self.data_dataframe['Shot'].to_numpy().copy()[1::self.frame_skip]
        if self.game_features_flag:
            skipped_dataframe['Target no'] = self.data_dataframe['Target no'].to_numpy().copy()[1::self.frame_skip]
        self.data_dataframe = skipped_dataframe

    @property
    def discretized_x(self):
        return self.discretize_x_function(self.data_dataframe['Delta X'].values)

    @property
    def discretized_y(self):
        return self.discretize_y_function(self.data_dataframe['Delta Y'].values)

    @property
    def image_paths(self):
        return self.data_dataframe['Image Path'].values

    @property
    def click_values(self):
        return self.data_dataframe['Shot'].values

    @property
    def number_of_targets(self):
        return self.data_dataframe['Target no'].values


if __name__ == "__main__":
    normalizer = DataNormalizer('')
            
