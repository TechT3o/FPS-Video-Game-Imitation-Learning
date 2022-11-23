from data_recording.data_recorder import DataRecorder
from win32api import GetSystemMetrics

# WINDOW_COORDINATES = (0, 0, 1920, 1080)
WINDOW_COORDINATES = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
RESET_CURSOR_FLAG = False
SAVE_PATH = ''
# print(win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))

data_recorder = DataRecorder(save_path=SAVE_PATH, window_coordinates=WINDOW_COORDINATES,
                             reset_cursor_flag=RESET_CURSOR_FLAG)  # replace with your own directory
data_recorder.run()