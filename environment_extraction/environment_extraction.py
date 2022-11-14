import cv2
import numpy as np
import mss
import time

import pyautogui
import pydirectinput
import win32api
import win32con

def nothing(x):
    pass
def click(x,y):
    #win32api.SetCursorPos((x, y))
    pydirectinput.moveTo(x, y)
    #pydirectinput.moveRel(x-AIMLABS_CENTER[0], y-AIMLABS_CENTER[1])
    # nx = int(x * 65535 / win32api.GetSystemMetrics(0))
    # ny = int(y * 65535 / win32api.GetSystemMetrics(1))
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(x/1920*65535.0), int(y/1080*65535.0))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(0.5)
    # pydirectinput.moveRel(AIMLABS_CENTER[0], AIMLABS_CENTER[1])

AIMLABS_CENTER = 960, 523
for i in range(4):
    time.sleep(1)
    print(3-i)

for kappa in range(30*2):

    time_1 = cv2.getTickCount()
    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        img = np.array(sct.grab(monitor))

    k = cv2.waitKey(1) & 0xFF
    if k == ord("c"):
        break

    time_2 = cv2.getTickCount()
    fps = (time_2 - time_1) / cv2.getTickFrequency()
    #print(int(1/fps))
    # hard coded hsv values that give the blue color of Aimlabs circles. Found using color_selection_tool.py
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([82, 130, 10])
    upper_color = np.array([96, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, (15, 15))
    mask = cv2.dilate(mask, (5, 5))
    mask = cv2.dilate(mask, (5, 5))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for contour in contours:
    # rect = cv.minAreaRect(contours[0])
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    #img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    centers = []
    for contour in contours:
        moment = cv2.moments(contour)
        x = int(moment["m10"] / (moment["m00"]+ 1e-5))
        y = int(moment["m01"] / (moment["m00"]+ 1e-5))
        if moment["m00"] < 200:
            continue
        img = cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
        img = cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=2)
        centers.append((x, y))
        # pydirectinput.moveTo(x, y)

    if len(centers) > 0:
        # pydirectinput.doubleClick(centers[0][0], centers[0][1])
        print(f'Center of blobs position {centers[0][0] + 5, centers[0][1] + 10}')
        print(pyautogui.position())
        click(centers[0][0]+5, centers[0][1])
        # pydirectinput.moveTo(200, 0)


    res = cv2.bitwise_and(img, img, mask=mask)
    # edges = cv2.Canny(res,minval,maxval,True)
    cv2.imshow('image', cv2.resize(img, (800, 600)))
cv2.destroyAllWindows()