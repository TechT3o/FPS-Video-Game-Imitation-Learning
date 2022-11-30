import numpy as np
from typing import Tuple
import win32api
import win32con
import json
import time
import os
import cv2


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


def mouse_action(x_motion, y_motion, click, shoot_wait_time: float = 0.1) -> None:
    """
    Function that moves mouse to target (width,height) coordinates and clicks
    :param x_motion: mouse motion to be done in the x coordinate
    :param y_motion: mouse motion to be done in the y coordinate
    :param click: whether to click the mouse
    :param shoot_wait_time: float of time to wait after shooting
    :return: None
    """

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_motion, y_motion, 0, 0)

    if click:
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


def check_and_create_directory(path_to_save: str) -> None:
    """
    Checks if directory exists and if not it creates it
    :param path_to_save: path of directory
    :return: None
    """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)


def json_to_dict(path: str):
    """
    loads json file to a dictionary
    :param path: json file path
    :return: None
    """
    with open(path) as json_file:
        data = json.load(json_file)
        return data


def preprocess_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    preprocesses the image to make it in a form to input to the model
    :param image: image to be processed
    :param image_size: size to reshape image in
    :return: processed image
    """
    image = image / 255.
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
    return image