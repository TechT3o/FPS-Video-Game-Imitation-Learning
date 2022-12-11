"""
Run this script to extract the number of targets in the sphere clash game mode of 3D aim trainer
by color filtering and finding the correct contours. The target no information is then added to the recorded data
in the csvs.
"""
from environment_extracting.environment_extraction import EnvironmentExtractor
from agent_training.data_normalizer import DataNormalizer
import cv2
import os
import pandas as pd

DATA_PATH = "C:\\Users\\thpap\\Dropbox\\Data_New"

env_extractor = EnvironmentExtractor()
data_normalizer = DataNormalizer(DATA_PATH)
data_normalizer.data_dataframe = pd.DataFrame()
data_normalizer.load_csvs()
target_number_list = []

for image_path in data_normalizer.image_paths:
    img = cv2.imread(os.path.join(DATA_PATH, image_path.split("/")[-1]))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    env_extractor.frame = img
    env_extractor.color_filtering()
    env_extractor.find_targets(visualize=False)
    # print(env_extractor.number_of_targets)
    target_number_list.append(env_extractor.number_of_targets)
    env_extractor.clear_targets()

data_normalizer.data_dataframe["Target no"] = target_number_list
data_normalizer.data_dataframe.to_csv('csv_w_targets.csv', mode='w')

