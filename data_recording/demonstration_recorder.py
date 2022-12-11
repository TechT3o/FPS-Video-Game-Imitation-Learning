from data_recording.data_recorder import DataRecorder
import time
import win32api
from statics import check_and_create_directory, start_countdown
import os
from cv2 import imwrite
import csv


class DemoRecorder(DataRecorder):
    def __init__(self, window_coordinates, reset_cursor_flag, save_path):
        self.demo_path = save_path
        check_and_create_directory(self.demo_path)
        super(DemoRecorder, self).__init__(window_coordinates, reset_cursor_flag, self.demo_path)

    def record_demo(self) -> None:
        """
        main function that runs to record the data
        :return: None
        """
        ltime = time.localtime(time.time())
        timestamp = f'{ltime.tm_year}_{ltime.tm_mon}_{ltime.tm_mday}_{ltime.tm_hour}_{ltime.tm_min}_{ltime.tm_sec}'
        file_path = os.path.join(self.csv_path, f'data_{timestamp}.csv')
        current_frames_path = os.path.join(self.frames_path, f'recording_{timestamp}')
        check_and_create_directory(current_frames_path)

        with open(file_path, 'w+', newline='') as csv_file:
            data_writer = csv.writer(csv_file)
            data_writer.writerow(["Image Path", "Delta X", "Delta Y", "Shot", "Hit Edge Flag"])
            frame_index = 1
            start_countdown(4)
            while True:
                loop_start_time = time.time()
                self.mouse_logger.get_mouse_states()
                # print(self.mouse_logger.d_x, self.mouse_logger.d_y)
                image_save_path = os.path.join(current_frames_path, f'Frame_{timestamp}_{frame_index}.jpg')
                imwrite(image_save_path, self.environment.get_image())
                data_writer.writerow([os.path.join(os.path.join('../data', 'frames'),
                                                   os.path.join(f'recording_{timestamp}',
                                                                f'Frame_{timestamp}_{frame_index}.jpg')),
                                      self.mouse_logger.d_x, self.mouse_logger.d_y, self.mouse_logger.l_click,
                                      self.mouse_logger.hit_edge()])
                frame_index += 1
                while time.time() < loop_start_time + 1/self.fps:
                    pass

                if win32api.GetAsyncKeyState(ord('Q'))&0x0001 > 0:
                    print('Quit')
                    break

    @property
    def demo_folder_path(self):
        return self.demo_path
