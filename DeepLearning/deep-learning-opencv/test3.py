__author__ = 'Umesh'


import numpy as np
import cv2
import os

def read_image(file1_in):
    """

    :param file1_in:
    :param file1_out:
    :return:
    """
    im = cv2.imread(file1_in)
    print(type(im))
    print(im.shape)
    cv2.imshow('gray_scale',im)
    cv2.waitKey(0)

if __name__ == '__main__':

    imagePath1 = "messi.jpg"
    read_image(imagePath1)