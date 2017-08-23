__author__ = 'Umesh'

import numpy as np
import cv2

image1 = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images\118_case769_front_Top_Bottom_0.jpg"
image2 = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images\118_case769_front_Top_Bottom_-1.jpg"

im1 = cv2.imread(image1)
im2 = cv2.imread(image2)
im3 = im1 - im2
cv2.imshow('diff', im3)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgmask = fgbg.apply(cap2)
# cv2.imshow('frame',fgmask)
k = cv2.waitKey(0)
cv2.destroyAllWindows()