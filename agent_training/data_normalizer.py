import os
import numpy as np
import pandas as pd
from singleton import Singleton


class DataNormalizer(metaclass=Singleton):
    def __init__(self, data_path: str = ''):

        self.csv_path = os.path.join(data_path, os.path.join('data', 'csvs'))
        self.action_space_x = np.array([-300, -200, -150, -100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 50, 100, 150, 200, 300])
        self.action_space_y = np.array([-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100])

        self.discretize_x_function = np.vectorize(lambda x: self.action_space_x[(np.abs(self.action_space_x - x)).argmin()])
        self.discretize_y_function = np.vectorize(lambda y: self.action_space_y[(np.abs(self.action_space_y - y)).argmin()])
        self.data_dataframe = pd.DataFrame()
        self.load_csvs()
        # print(self.data_dataframe)
        self.keep_non_edge_data()
        self.data_dataframe['Delta X'] = self.discretize_x_function(self.data_dataframe['Delta X'].values)
        self.data_dataframe['Delta Y'] = self.discretize_y_function(self.data_dataframe['Delta Y'].values)
        self.one_hot_encoding()

    def one_hot_encoding(self):
        one_hot_dataframe_x = pd.get_dummies(self.data_dataframe['Delta X']).to_numpy()
        one_hot_dataframe_y = pd.get_dummies(self.data_dataframe['Delta Y']).to_numpy()
        one_hot_dataframe_click = pd.get_dummies(self.data_dataframe['Shot']).to_numpy()

        return one_hot_dataframe_x, one_hot_dataframe_y, one_hot_dataframe_click

    def load_csvs(self):
        print("Loading csvs")
        for csv_file in os.listdir(self.csv_path):
            sample_dataframe = pd.read_csv(os.path.join(self.csv_path, csv_file))
            self.data_dataframe = pd.concat([self.data_dataframe, sample_dataframe], axis=0, ignore_index=True)
            #print(self.data_dataframe)

    def keep_non_edge_data(self):
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


if __name__ == "__main__":
    normalizer = DataNormalizer('')
# print(normalizer.discretized_x, normalizer.discretized_y)
            
