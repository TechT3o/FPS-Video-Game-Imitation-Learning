from singleton import Singleton
from statics import json_to_dict
import os


class Parameters(metaclass=Singleton):
    def __init__(self):
        self.parameter_dict = json_to_dict(os.path.join('agent_training', 'model_params.json'))

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
