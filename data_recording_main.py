"""
Records screen data and mouse movement inside hte window coordinates and saves the information in csv files
in the save path provided. If the game being recorded resets the cursor in between frames set reset cursor flag
to true
"""

from data_recording.data_recorder import DataRecorder
from statics import start_countdown
from win32api import GetSystemMetrics

# WINDOW_COORDINATES = (0, 0, 1920, 1080)
WINDOW_COORDINATES = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
RESET_CURSOR_FLAG = True
SAVE_PATH = ''


data_recorder = DataRecorder(save_path=SAVE_PATH, window_coordinates=WINDOW_COORDINATES,
                             reset_cursor_flag=RESET_CURSOR_FLAG)  # replace with your own directory
start_countdown(7)
data_recorder.run()
