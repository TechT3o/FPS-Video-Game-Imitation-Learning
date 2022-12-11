from singleton import Singleton
from statics import json_to_dict
import os


class Parameters(metaclass=Singleton):
    """
    Singleton class that reads parameters from a json files and is called from other classes to access the parameters
    """
    def __init__(self):
        try:
            self.parameter_dict = json_to_dict(os.path.join('agent_training', 'model_params.json'))
        except FileNotFoundError:
            self.parameter_dict = json_to_dict('model_params.json')

    @property
    def save_path(self):
        return self.parameter_dict['save_path']

    @property
    def data_path(self):
        return self.parameter_dict['data_path']

    @property
    def model_base(self):
        return self.parameter_dict['model_base']

    @property
    def lstm_flag(self):
        return self.parameter_dict['lstm_flag']

    @property
    def feature_chain_flag(self):
        return self.parameter_dict['feature_chain_flag']

    @property
    def augmentation(self):
        return self.parameter_dict['augmentation']

    @property
    def image_size_x(self):
        return self.parameter_dict['image_size_x']

    @property
    def image_size_y(self):
        return self.parameter_dict['image_size_y']

    @property
    def channel_size(self):
        return self.parameter_dict['channel_size']

    @property
    def time_steps(self):
        return self.parameter_dict['time_steps']

    @property
    def validation_fraction(self):
        return self.parameter_dict['validation_fraction']

    @property
    def test_fraction(self):
        return self.parameter_dict['test_fraction']

    @property
    def loading_flag(self):
        return self.parameter_dict['loading_flag']

    @property
    def batch_size(self):
        return self.parameter_dict['batch_size']

    @property
    def debias_shooting(self):
        return self.parameter_dict['debias_shooting']

    @property
    def one_shot_path(self):
        return self.parameter_dict['one_shot_path']

    @property
    def agent_path(self):
        return self.parameter_dict['agent_path']

    @property
    def dagger_path(self):
        return self.parameter_dict['dagger_path']

    @property
    def adjacent_label_encoding(self):
        return self.parameter_dict['adjacent_label_encoding']
