import os

from agent_training.model_training import ModelTrainer

SAVE_PATH = ''
DATA_PATH = ''
SAVE_PATH = os.getcwd() if SAVE_PATH == '' else SAVE_PATH
DATA_PATH = os.getcwd() if DATA_PATH == '' else DATA_PATH

trainer = ModelTrainer(data_path=DATA_PATH, model_base='MobileNetv3',
                       lstm_flag='LSTM', feature_chain_flag=False, time_steps=10)
