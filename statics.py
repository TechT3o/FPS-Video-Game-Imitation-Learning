import numpy as np
import os


def discretize(value, discretizer):
    discretized_value = discretizer[np.abs(discretizer - value).argmin()]
    return discretized_value


def check_and_create_directory(path_to_save):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)