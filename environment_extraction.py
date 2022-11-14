import cv2
import numpy as np
import mss
import time


def nothing():
    pass


for i in range(4):
    time.sleep(1)
    print(3-i)

# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('minval', 'image', 0, 1000, nothing)
cv2.createTrackbar('maxval', 'image', 0, 1000, nothing)

# for aimlabs blue pick (82,130,10)
cv2.createTrackbar('hue_l', 'image', 0, 179, nothing)
cv2.createTrackbar('sat_l', 'image', 0, 255, nothing)
cv2.createTrackbar('bri_l', 'image', 0, 255, nothing)

# for aimlabs blue pick (96,255,255)
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

    time_2 = cv2.getTickCount()
    fps = (time_2 - time_1) / cv2.getTickFrequency()
    print(int(1/fps))

    # get current positions of four trackbars
    minval = cv2.getTrackbarPos('minval', 'image')
    maxval = cv2.getTrackbarPos('maxval', 'image')
    a1 = cv2.getTrackbarPos('hue_l', 'image')
    a2 = cv2.getTrackbarPos('sat_l', 'image')
    a3 = cv2.getTrackbarPos('bri_l', 'image')

    b1 = cv2.getTrackbarPos('hue_h', 'image')
    b2 = cv2.getTrackbarPos('sat_h', 'image')
    b3 = cv2.getTrackbarPos('bri_h', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        cv2.imshow('image', cv2.resize(img, (800, 600)))
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_color = np.array([a1, a2, a3])
        upper_color = np.array([b1, b2, b3])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, (15, 15))
        mask = cv2.dilate(mask, (5, 5))
        mask = cv2.dilate(mask, (5, 5))
        contours = cv2.drawContours(mask)

        res = cv2.bitwise_and(img, img, mask=mask)
        # process mask
        # edges = cv2.Canny(res,minval,maxval,True)
        cv2.imshow('image', cv2.resize(res, (800, 600)))
cv2.destroyAllWindows()