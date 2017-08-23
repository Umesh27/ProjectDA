__author__ = 'Umesh'

import cv2, os
import numpy as np


def imageReadShow(fileName):
    img = cv2.imread(fileName, 0)
    cv2.imshow("image_show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imageReadColorShow(fileName):
    img = cv2.imread(fileName, cv2.IMREAD_COLOR)
    cv2.imshow("image_show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findContours(fileName):
    """

    :param fileName:
    :return:
    """

    im = cv2.imread(fileName)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)

    # Detect contours using both methods on the same image
    im2, contours1, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im2, contours1, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Copy over the original image to separate variables
    img1 = im.copy()
    #img2 = im.copy()
    cnt = contours1[5]
    # for i in range(len(contours1)):
    #     print(i)
    #     print((contours1[i]))

    #print(len(contours1), cnt)
    # Draw both contours onto the separate images
    #cv2.drawContours(img1, [cnt], 4, (255, 0, 0), 2)
    cv2.drawContours(im2, contours1, 4, (255,0,0), 2)
    cv2.drawContours(img1, contours1, -1, (255,0,0), 3)

    out = np.hstack([im, img1])
    # print(type(out))
    # print(out)
    # contours1_out = os.path.join(image_dir, "trainData_.csv")
    # print("Output path : ",contours1_out)
    # np.savetxt(contours1_out, out)#, delimiter=",")

    # Now show the image
    cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(os.getcwd())
    # image_dir = r"D:\Umesh\Learn_project\OpenCV\Study"
    image_dir = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images"
    # image_path1 = os.path.join(image_dir, "box_jpg.jpg")
    # image_path2 = os.path.join(image_dir, "box.png")
    # image_path3 = os.path.join(image_dir, "messi_new.jpg")
    image_path = os.path.join(image_dir, "118_case769_front_Top_Bottom_0.jpg")
    # function 1 to read and show the image
    #imageReadShow(image_path)
    #imageReadColorShow(image_path)
    findContours(image_path)