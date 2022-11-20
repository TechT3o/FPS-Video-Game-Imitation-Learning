import cv2
import pytesseract
import mss
import numpy as np


class OCR:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

        # Hard coded values to get score from Aim Labs
        self.y_l_score = 30
        self.y_h_score = 90
        self.x_l_score = 625
        self.x_h_score = 840

        self.score_history = []

    def get_score(self, image) -> None:
        score = pytesseract.image_to_string(image)
        if score != '':
            try:
                self.score_history.append(int(score))
            except Exception as e:
                print(e)

    def shot_or_missed(self) -> None:
        if len(self.score_history) > 1 and self.score_history[-1] > self.score_history[-2]:
            print('Shot target!')
        elif len(self.score_history) > 1 and self.score_history[-1] < self.score_history[-2]:
            print('Missed target')

    @property
    def score(self):
        return self.score_history[-1]
