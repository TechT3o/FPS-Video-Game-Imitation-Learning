from mouse_input import MouseLogger
from environment_extracting.environment_extraction import EnvironmentExtractor
import time


WINDOW_COORDINATES = (0, 0, 1920, 1080)


class DataRecorder:
    def __init__(self, save_path):
        self.mouse_logger = MouseLogger(window_coordinates=(1302, 588), reset_cursor_flag=True)
        self.environment = EnvironmentExtractor(WINDOW_COORDINATES)
        self.data_writer = None  #TODO put spencers' datawriter
        self.fps = 32

    def run(self):
        while True:
            loop_start_time = time.time()

            self.environment.get_image()
            self.mouse_logger.get_mouse_states()
            print(self.mouse_logger.d_x, self.mouse_logger.d_y)
            if self.mouse_logger.l_click:
                print('shot')
            while time.time() < loop_start_time + 1/self.fps:
                pass


if __name__ == "__main__":
    data_recorder = DataRecorder('')
    data_recorder.run()
