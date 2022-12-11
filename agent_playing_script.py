"""
Scripts that runs a trained agent.
"""

import tensorflow as tf
import numpy as np
from statics import start_countdown, mouse_action, preprocess_image
import time
import mss
from win32api import GetAsyncKeyState


class Agent:
    """
    class that loads an agent (keras model) captures the frames of the screen and applies the same preprocessing
     used in training, feeds this input to the agent to predict and performs the predicted action
    """
    def __init__(self, model_path: str):
        self.agent = tf.keras.models.load_model(model_path)
        self.time_steps = self.agent.input.shape[1]
        print(self.time_steps)
        print(self.agent.summary())
        self.ACTION_SPACE_X = np.array([-300, -200, -150, -100, -50, -25,
                                        -10, -5, -1, 0, 1, 5, 10, 50, 100, 150, 200, 300])
        self.ACTION_SPACE_Y = np.array([-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100])
        self.has_features = self.find_feature_chain()
        self.FPS = 15
        self.buffer = []
        self.action_threshold = 0.12
        self.shooting_threshold = 0.25

    def run_agent(self):
        while True:

            loop_start_time = time.time()
            with mss.mss() as sct:
                monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
                img = np.array(sct.grab(monitor))[:, :, :3]

            processed_image = preprocess_image(img, (240, 135)).reshape((240, 135, 3))
            self.buffer.append(processed_image)

            if len(self.buffer) >= self.time_steps:
                if self.has_features:
                    features, click_pred, mouse_x_pred, mouse_y_pred =\
                        self.agent.predict_on_batch(np.array([self.buffer]))
                else:
                    click_pred, mouse_x_pred, mouse_y_pred = self.agent.predict_on_batch(np.array([self.buffer]))

                self.buffer.pop(0)
                x_predictions = mouse_x_pred[0][-1]
                y_predictions = mouse_y_pred[0][-1]
                click_predictions = click_pred[0][-1]

                # print(x_predictions)
                # print(y_predictions)
                # print(click_predictions)

                x_motion = self.ACTION_SPACE_X[x_predictions.argmax()] if\
                    max(x_predictions) > self.action_threshold else 0
                y_motion = self.ACTION_SPACE_Y[y_predictions.argmax()] if\
                    max(y_predictions) > self.action_threshold else 0

                if len(click_predictions) > 1:
                    click = 1 if click_predictions[1] > self.shooting_threshold else 0
                else:
                    click = 1 if click_predictions > self.shooting_threshold else 0

                mouse_action(x_motion, y_motion, click, 0.01)

            if GetAsyncKeyState(ord('Q')) & 0x0001 > 0:
                print('Quit')
                break

            while time.time() < loop_start_time + 1 / self.FPS:
                pass
            print(1 / (time.time() - loop_start_time))

    def find_feature_chain(self) -> bool:
        """
        Finds whether the trained model had a feature chain
        :return: True if it did, False otherwise
        """
        for layer in self.agent.layers:
            if layer.name == 'features_out':
                return True
        return False


if __name__ == "__main__":
    agent = Agent('agents\\agent_tile_5\\agent.h5')
    start_countdown(6)
    agent.run_agent()
