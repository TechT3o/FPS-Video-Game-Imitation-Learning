from mouse_input import MouseLogger
from environment_extracting.environment_extraction import EnvironmentExtractor
import time
import csv

WINDOW_COORDINATES = (0, 0, 1920, 1080)


class DataRecorder:
    """
    Class that records the user input and the image frames and saves that information in csv files
    """
    def __init__(self, save_path):
        """
        class constructor
        :param save_path: path where the data should be saved
        """
        self.mouse_logger = MouseLogger(window_coordinates=(1302, 588), reset_cursor_flag=True)
        self.environment = EnvironmentExtractor(WINDOW_COORDINATES)
        ltime = time.localtime(time.time())
        self.csvfile = open(f"{save_path}\data_{ltime.tm_year}_{ltime.tm_mon}_{ltime.tm_mday}_{ltime.tm_hour}_{ltime.tm_min}_{ltime.tm_sec}.csv", 'w', newline='') #TODO: close file after loop if not exited via Ctrl+C
        self.data_writer = csv.writer(self.csvfile)
        self.data_writer.writerow(["Image Path", "Start X", "Start Y", "End X", "End Y", "Shot"])
        self.fps = 30

    def run(self) -> None:
        """
        main function that runs to record hte data
        :return: None
        """
        prev_x = WINDOW_COORDINATES[2] / 2
        prev_y = WINDOW_COORDINATES[3] / 2
        while True:
            loop_start_time = time.time()
            self.mouse_logger.get_mouse_states()
            #print(self.mouse_logger.d_x, self.mouse_logger.d_y)
            self.data_writer.writerow([self.environment.get_image(), prev_x, prev_y, self.mouse_logger.d_x, self.mouse_logger.d_y, self.mouse_logger.l_click])
            prev_x = self.mouse_logger.d_x
            prev_y = self.mouse_logger.d_y
            while time.time() < loop_start_time + 1/self.fps:
                pass


if __name__ == "__main__":
    data_recorder = DataRecorder(r'D:\UCLA\Fall 2022\209AS\Project\Data') # replace with your own directory
    data_recorder.run()
