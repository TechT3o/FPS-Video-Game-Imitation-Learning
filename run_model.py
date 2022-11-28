import tensorflow as tf
from tensorflow import keras
from win32api import GetSystemMetrics
from agent_training.data_preprocessing import DataProcessor
from environment_extracting.environment_extraction import EnvironmentExtractor

def main():
    model_path = r"D:\UCLA\Fall2022\209AS\Project\Video-games-target-generalization\models\model.h5"
    model = keras.models.load_model(model_path)
    model.summary()

    data_processor = DataProcessor()
    environment_extractor = EnvironmentExtractor((0, 0, GetSystemMetrics(0), GetSystemMetrics(1)))

    img = data_processor.preprocess_image(environment_extractor.get_image())
    model.predict(img)
    #TODO: act based on model predictions

def main_test():
    model_path = r"D:\UCLA\Fall2022\209AS\Project\Video-games-target-generalization\models\model.h5"
    model = keras.models.load_model(model_path)
    model.summary()

    data_processor = DataProcessor()

    # hard-coding image path to test--take screenshot during live run
    img = data_processor.get_image(r"D:\UCLA\Fall2022\209AS\Project\Data\data\frames\recording_2022_11_24_14_14_38\Frame_2022_11_24_14_14_38_412.jpg")
    img = data_processor.preprocess_image(img)

    model.predict(img)

if __name__ == "__main__":
    main_test()
