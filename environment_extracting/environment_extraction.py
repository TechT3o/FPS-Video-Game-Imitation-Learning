from environment_extracting.ocr import OCR
from typing import Tuple, List
import mss
import numpy as np
import cv2


class EnvironmentExtractor:
    """
    Class that extracts visual information from the environment
    """
    ocr: OCR
    screen_coordinates: Tuple[int, int, int, int]
    frame: np.ndarray
    target_mask: np.ndarray
    target_centers: List

    def __init__(self, window_capture_coordinates: Tuple[int, int, int, int] = (0, 0, 1920, 1080)):
        """"""
        self.ocr = OCR()
        self.screen_coordinates = window_capture_coordinates
        self.frame = self.get_image()
        self.target_mask = np.array([])
        self.target_centers = []

        self.lower_color = np.array([10, 78, 209])
        self.upper_color = np.array([21, 115, 255])
        self.AREA = 40

        # 3d aimtrainer color
        # self.lower_color = np.array([9, 81, 255])
        # self.upper_color = np.array([28, 154, 255])
        # self.AREA = 40

        # Aim Lab color
        # self.lower_color = np.array([82, 130, 10])
        # self.upper_color = np.array([96, 255, 255])
        # self.AREA = 500

    def get_image(self) -> np.ndarray:
        """
        Gets screenshot for the given window coordinates
        :return: array of the captured image
        """
        with mss.mss() as sct:
            monitor = {"top": self.screen_coordinates[0], "left": self.screen_coordinates[1],
                       "width": self.screen_coordinates[2], "height": self.screen_coordinates[3]}
            return np.array(sct.grab(monitor))

    def get_ocr_image(self) -> np.ndarray:
        """
        Gets screenshot for the coordinates given by OCR
        :return: array of the captured image
        """
        with mss.mss() as sct:
            monitor = {"top": self.ocr.y_l_score, "left": self.ocr.x_l_score,
                       "width": self.ocr.x_h_score-self.ocr.x_l_score, "height": self.ocr.y_h_score-self.ocr.y_l_score}
            return np.array(sct.grab(monitor))

    def color_filtering(self) -> None:
        """
        Filters the image based on the color and produces a mask
        :return: None
        """
        # hard coded hsv values that give the blue color of Aimlabs circles. Found using color_selection_tool.py
        hsv = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        # manipulates mask to make smoother
        mask = cv2.erode(mask, (15, 15))
        mask = cv2.dilate(mask, (5, 5))
        self.target_mask = cv2.dilate(mask, (5, 5))

    def find_targets(self, visualize: bool = False) -> None:
        """
        Finds centers of contours of the mask which has the color filtered objects
        :return: None
        """
        contours, hierarchy = cv2.findContours(self.target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moment = cv2.moments(contour)
            # add to avoid divide by 0 error
            x = int(moment["m10"] / (moment["m00"] + 1e-5))
            y = int(moment["m01"] / (moment["m00"] + 1e-5))
            # filter by area size
            if moment["m00"] < self.AREA:
                continue
            if visualize:
                cv2.drawContours(self.frame, contour, -1, (0, 0, 255), 2)
                cv2.circle(self.frame, (x, y), radius=1, color=(0, 0, 255), thickness=2)
            self.target_centers.append((x, y))
        if visualize:
            cv2.imshow('contoured img', cv2.resize(self.frame, (240*4, 135*4)))
            cv2.waitKey(1)

    def clear_targets(self):
        self.target_centers = []

    def test_extractor(self):
        while True:
            self.frame = self.get_image()
            self.color_filtering()
            self.find_targets(visualize=True)
            print(self.number_of_targets)
            self.clear_targets()
        # cv2.destroyAllWindows()

    @property
    def number_of_targets(self):
        return len(self.target_centers)

    @property
    def score_history(self):
        return self.ocr.score_history


if __name__ == "__main__":
    env_ext = EnvironmentExtractor((0, 0, 1920, 1080))
    env_ext.test_extractor()
