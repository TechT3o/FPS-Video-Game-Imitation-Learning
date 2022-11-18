from ocr import OCR
import mss
import numpy as np
import cv2


class EnvironmentExtractor:
    def __init__(self, window_capture_coordinates):
        self.ocr = OCR()
        self.screen_coordinates = window_capture_coordinates
        self.frame = self.get_image()
        self.target_mask = np.array([])
        self.target_centers = []

    def get_image(self) -> np.ndarray:
        with mss.mss() as sct:
            monitor = {"top": self.screen_coordinates[0], "left": self.screen_coordinates[1],
                       "width": self.screen_coordinates[2], "height": self.screen_coordinates[3]}
            return np.array(sct.grab(monitor))

    def get_ocr_image(self) -> np.ndarray:
        with mss.mss() as sct:
            monitor = {"top": self.ocr.y_l_score, "left": self.ocr.x_l_score,
                       "width": self.ocr.x_h_score-self.ocr.x_l_score, "height": self.ocr.y_h_score-self.ocr.y_l_score}
            return np.array(sct.grab(monitor))

    def color_filtering(self) -> None:
        # hard coded hsv values that give the blue color of Aimlabs circles. Found using color_selection_tool.py
        hsv = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2HSV)
        lower_color = np.array([82, 130, 10])
        upper_color = np.array([96, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # manipulates mask to make smoother
        mask = cv2.erode(mask, (15, 15))
        mask = cv2.dilate(mask, (5, 5))
        self.target_mask = cv2.dilate(mask, (5, 5))

    def find_targets(self) -> None:
        contours, hierarchy = cv2.findContours(self.target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            self.target_centers.append((x, y))

    @property
    def number_of_targets(self):
        return len(self.target_centers)

    @property
    def score_history(self):
        return self.ocr.score_history
