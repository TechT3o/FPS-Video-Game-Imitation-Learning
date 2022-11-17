import cv2
import numpy as np
import mss
import time
import win32api
import win32con
from typing import Tuple


def mouse_click_at(target_coordinates: Tuple[int, int], center_coordinates: Tuple[int, int],
                   shoot_wait_time: float = 0.1) -> None:
    """
    Function that moves mouse to target (width,height) coordinates and clicks
    :param target_coordinates: tuple with (width,height) target coordinates where the mouse will click
    :param center_coordinates: tuple with (width,height) central coordinates relative to which the motion will occur
    :param shoot_wait_time: float of time to wait after shooting
    :return: None
    """

    ox, oy = win32api.GetCursorPos()
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, target_coordinates[0] - center_coordinates[0],
    #                      target_coordinates[1] - center_coordinates[1], 0, 0)
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


AIMLABS_CENTER = (960, 539)  # will probably remove as cursor always locke at center, so we can grab it from win32 api
SHOOTING_WAIT = 0.1
SECONDS_TO_RUN = 30


class EnvironmentExtractor:
    def __init__(self):
        self.ocr = 0


if __name__ == "__main__":
    start_countdown(5)
    for kappa in range(int((1/SHOOTING_WAIT))*SECONDS_TO_RUN):

        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            img = np.array(sct.grab(monitor))

        # hard coded hsv values that give the blue color of Aimlabs circles. Found using color_selection_tool.py
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        lower_color = np.array([82, 130, 10])
        upper_color = np.array([96, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # manipulates mask to make smoother
        mask = cv2.erode(mask, (15, 15))
        mask = cv2.dilate(mask, (5, 5))
        mask = cv2.dilate(mask, (5, 5))

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            moment = cv2.moments(contour)
            # add to avoid divide by 0 error
            x = int(moment["m10"] / (moment["m00"] + 1e-5))
            y = int(moment["m01"] / (moment["m00"] + 1e-5))
            # filter by area size
            if moment["m00"] < 500:
                continue
            # Uncomment to illustrate found targets on image
            # cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
            # cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=2)
            centers.append((x, y))

        if len(centers) > 0:
            mouse_click_at(centers[0], AIMLABS_CENTER, SHOOTING_WAIT)

        # cv2.imshow('image', cv2.resize(img, (800, 600)))
    # cv2.destroyAllWindows()
