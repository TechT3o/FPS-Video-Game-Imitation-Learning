import numpy as np
from typing import Tuple
import win32api
import win32con
import json
import time
import os


def mouse_click_at(target_coordinates: Tuple[int, int], shoot_wait_time: float = 0.1) -> None:
    """
    Function that moves mouse to target (width,height) coordinates and clicks
    :param target_coordinates: tuple with (width,height) target coordinates where the mouse will click
    :param shoot_wait_time: float of time to wait after shooting
    :return: None
    """

    ox, oy = win32api.GetCursorPos()
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, target_coordinates[0] - ox, target_coordinates[1] - oy, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(shoot_wait_time)


def start_countdown(countdown_number: int) -> None:
    """
    counts down before starting the program to give time to open the game window
    :param countdown_number: seconds to wait
    :return: None
    """
    for i in range(countdown_number):
        print(countdown_number - i)
        time.sleep(1)


def discretize(value, discretizer):
    discretized_value = discretizer[np.abs(discretizer - value).argmin()]
    return discretized_value


def check_and_create_directory(path_to_save):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)


def json_to_dict(path: str):
    with open(path) as json_file:
        data = json.load(json_file)
        return data
