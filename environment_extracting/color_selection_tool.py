"""
Gui that shows you your screen and has sliders to change the HSV threshold values to do color selection. Useful website
for selecting the color is https://programmingdesignsystems.com/color/color-models-and-color-spaces/index.html
"""
import cv2
import numpy as np
import mss


def nothing() -> None:
    """
    Does nothing and is used as callback to create Trackbar
    :return: None
    """
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')
# create trackbars for color change

# hsv for 3daimtrainer is 9, 81, 255 and 28, 154, 255

# for Aimlabs blue pick (82,130,10)
cv2.createTrackbar('hue_l', 'image', 0, 179, nothing)
cv2.createTrackbar('sat_l', 'image', 0, 255, nothing)
cv2.createTrackbar('bri_l', 'image', 0, 255, nothing)

# for Aimlabs blue pick (96,255,255)
cv2.createTrackbar('hue_h', 'image', 0, 179, nothing)
cv2.createTrackbar('sat_h', 'image', 0, 255, nothing)
cv2.createTrackbar('bri_h', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while 1:

    time_1 = cv2.getTickCount()
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        img = np.array(sct.grab(monitor))

    k = cv2.waitKey(1) & 0xFF
    if k == ord("c"):
        break

    # get current positions of trackbars
    hue_l = cv2.getTrackbarPos('hue_l', 'image')
    sat_l = cv2.getTrackbarPos('sat_l', 'image')
    bri_l = cv2.getTrackbarPos('bri_l', 'image')

    hue_h = cv2.getTrackbarPos('hue_h', 'image')
    sat_h = cv2.getTrackbarPos('sat_h', 'image')
    bri_h = cv2.getTrackbarPos('bri_h', 'image')
    on_off = cv2.getTrackbarPos(switch, 'image')

    if on_off == 0:
        cv2.imshow('image', cv2.resize(img, (800, 600)))
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_color = np.array([hue_l, sat_l, bri_l])
        upper_color = np.array([hue_h, sat_h, bri_h])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        res = cv2.bitwise_and(img, img, mask=mask)

        # print fps
        time_2 = cv2.getTickCount()
        fps = (time_2 - time_1) / cv2.getTickFrequency()
        print(int(1 / fps))

        cv2.imshow('image', cv2.resize(res, (800, 600)))
cv2.destroyAllWindows()
