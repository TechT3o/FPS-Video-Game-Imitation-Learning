from agent_training.model_training import ModelTrainer
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

trainer = ModelTrainer()
