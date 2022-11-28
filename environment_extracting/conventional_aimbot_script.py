import cv2
import numpy as np
import mss
from statics import start_countdown, mouse_click_at

SHOOTING_WAIT = 0.05
SECONDS_TO_RUN = 30

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
            mouse_click_at(centers[0], SHOOTING_WAIT)

        # cv2.imshow('image', cv2.resize(img, (800, 600)))
    # cv2.destroyAllWindows()
