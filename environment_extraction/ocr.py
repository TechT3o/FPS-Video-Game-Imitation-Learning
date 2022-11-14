import cv2
import pytesseract
import mss
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'


y_l_score = 30
y_h_score = 90
x_l_score = 625
x_h_score = 840

score_history = []

while 1:

    with mss.mss() as sct:
        monitor = {"top": y_l_score, "left": x_l_score, "width": x_h_score-x_l_score, "height": y_h_score-y_l_score}
        img = np.array(sct.grab(monitor))

    cv2.imshow('image', cv2.resize(img, (800, 600)))
    k = cv2.waitKey(1) & 0xFF


    score = pytesseract.image_to_string(img)

    if score != '':
        try:
            score_history.append(int(score))
        except Exception as e:
            print(e)

        if len(score_history) > 1 and score_history[-1] > score_history[-2]:
            print('Shot target!')
        elif len(score_history) > 1 and score_history[-1] < score_history[-2]:
            print('Missed target')

    if k == ord('e'):
        break

print(score_history)
cv2.destroyAllWindows()
