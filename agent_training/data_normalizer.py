import os
import numpy as np
import pandas as pd
from singleton import Singleton
from statics import visualize_labels


class DataNormalizer(metaclass=Singleton):
    """
    class that reads the csv data gets the image paths and cleans and discretizes them to a form used to be classified
    """
    def __init__(self, data_path: str = '', game_feature_chain: int = 0):
        """
        class constructor
        :param data_path: path where the csv files are stored
        """
        self.game_features_flag = game_feature_chain
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
        self.data_dataframe['Delta X'] = self.discretize_x_function(self.data_dataframe['Delta X'].values)
        self.data_dataframe['Delta Y'] = self.discretize_y_function(self.data_dataframe['Delta Y'].values)
        # self.one_hot_encoding()

    def one_hot_encoding_experimental(self):
        """
        Encodes the available actions in one hot form to be used as classification labels
        :return: numpy arrays with the one-hot encoding for the x_motion, y_motion, click actions
        """
        x = self.data_dataframe['Delta X'].to_numpy()
        y = self.data_dataframe['Delta Y'].to_numpy()
        one_hot_x = self.one_hot_encode_x(x)
        one_hot_y = self.one_hot_encode_y(y)
        one_hot_click = pd.get_dummies(self.data_dataframe['Shot']).to_numpy()

        if self.game_features_flag:
            one_hot_features = pd.get_dummies(self.data_dataframe['Target no']).to_numpy()
            return one_hot_x, one_hot_y, one_hot_click, one_hot_features

        return one_hot_x, one_hot_y, one_hot_click

    def one_hot_encoding(self):
        """
        Encodes the available actions in one hot form to be used as classification labels
        :return: numpy arrays with the one-hot encoding for the x_motion, y_motion, click actions
        """
        one_hot_x = pd.get_dummies(self.data_dataframe['Delta X']).to_numpy()
        one_hot_y = pd.get_dummies(self.data_dataframe['Delta Y']).to_numpy()
        one_hot_click = pd.get_dummies(self.data_dataframe['Shot']).to_numpy()

        visualize_labels(one_hot_x)
        visualize_labels(one_hot_y)
        visualize_labels(one_hot_click)

        if self.game_features_flag:
            one_hot_features = pd.get_dummies(self.data_dataframe['Target no']).to_numpy()
            return one_hot_x, one_hot_y, one_hot_click, one_hot_features

        return one_hot_x, one_hot_y, one_hot_click

    def load_csvs(self):
        """
        Loads the csv files in a pandas dataframe
        :return: None
        """
        if self.game_features_flag:
            self.data_dataframe = pd.read_csv(os.path.join(self.csv_path, "csv_targets - Copy.csv"))
        else:
            csvs = os.listdir(self.csv_path)
            csvs.remove("csv_targets - Copy.csv")

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
            
