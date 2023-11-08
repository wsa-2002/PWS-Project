import cv2
import pytesseract

import matplotlib.pyplot as plt

from . import Point


class Image:
    def __init__(self, path):
        self.img = cv2.imread(path)

    def show(self, img=None):
        if img is None:
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def crop(self, from_point: Point, to: Point, show=True):
        crop_img = self.img[from_point.y:to.y, from_point.x:to.x]
        if show:
            self.show(crop_img)
        return crop_img

    @staticmethod
    def detect_section(img):
        return pytesseract.image_to_string(img, config='--psm 6').replace('\n', '')
