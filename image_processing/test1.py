__author__ = 'Umesh'

# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2.cv2 as cv2
import os
import numpy as np
from PIL import Image, ImageChops
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True, help="first input image")
# ap.add_argument("-s", "--second", required=True, help="second")
# args = vars(ap.parse_args())


def get_diff(imPath1, imPath2):

    parentDir = os.path.split(imPath1)[0]
    # load the two input images
    imageA = cv2.imread(imPath1)
    imageB = cv2.imread(imPath2)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff1) = compare_ssim(grayA, grayB, full=True)
    diff = (diff1 * 255).astype("uint8")
    print("type of grayA", type(grayA))
    print("type of diff", type(diff1))

    outcsv = os.path.join(parentDir, "out_diff.csv")
    np.savetxt(outcsv, diff, delimiter=',')
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # print(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        print(x, y, w, h)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    # cv2.imshow("Original", imageA)
    # cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    #cv2.imshow("image_from_array", image_from_array)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

def get_diff1(imPath1, imPath2):

    # convert the images to grayscale
    im1 = Image.open(imPath1)
    im2 = Image.open(imPath2)
    print("Image open return type :", type(im1))

    diff = ImageChops.difference(im1, im1)
    print(diff)
    diff1 = np.array(diff)
    imageA = cv2.imread(imPath1)
    print("imread return type :", type(imageA))
    print("diff type :", type(diff1))
    cv2.imshow('diff1', diff1)
    cv2.waitKey(0)

    diff.show()

def get_diff2(file1, file2):
    """

    :param file1:
    :param file2:
    :return:
    """

    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)

    # convert the images to grayscale
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff1) = compare_ssim(grayA, grayB, full=True)
    diff = (diff1 * 255).astype("uint8")

    cv2.imshow('img1', grayA)
    cv2.imshow('img2', grayB)
    cv2.imshow('diff', diff)
    # print(grayA)
    # print(diff)
    (score2, diff2) = compare_ssim(grayA, diff, full=True)
    cv2.imshow('diff2', diff2)

    cv2.waitKey(0)

def get_outline(file1):
    """
    :param file1:
    :return:
    """
    im = cv2.imread(file1)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)

    # Detect contours using both methods on the same image
    im2, contours1, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Copy over the original image to separate variables
    img1 = im.copy()

    # Draw both contours onto the separate images
    cv2.drawContours(img1, contours1, 4, (255,0,0), 3)

    out = np.hstack([im, img1])

    # Now show the image
    cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_image(file1_in, file1_out):
    """

    :param file1_in:
    :param file1_out:
    :return:
    """
    im_gray = cv2.imread(file1_in, cv2.IMREAD_GRAYSCALE)#cv2.CV_LOAD_IMAGE_GRAYSCALE)
    outcsv = os.path.join(os.path.split(file1_out)[0], "grayScale_image.csv")
    np.savetxt(outcsv, im_gray, delimiter=',')

    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    outcsv2 = os.path.join(os.path.split(file1_out)[0], "bw_image.csv")
    np.savetxt(outcsv2, im_bw, delimiter=',')
    cv2.imwrite(file1_out, im_bw)

    cv2.imshow('gray_scale',im_gray)
    cv2.imshow('bw',im_bw)
    cv2.waitKey(0)

def read_image2(file1_in, file1_out):
    """

    :param file1_in:
    :param file1_out:
    :return:
    """
    im_gray = cv2.imread(file1_in, cv2.IMREAD_GRAYSCALE)#cv2.CV_LOAD_IMAGE_GRAYSCALE)
    outcsv = os.path.join(os.path.split(file1_out)[0], "grayScale_image.csv")
    np.savetxt(outcsv, im_gray, delimiter=',')
    ret,thresh = cv2.threshold(im_gray,127,255,0)

    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    #shapeMask = cv2.inRange(im_gray, lower, upper)
    # find the contours in the mask
    im2, contours1, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("I found %d black shapes" % (len(cnts)))
    # cv2.imshow("Mask", im_gray)

    # cv2.drawContours(im_gray, contours1, -1, (0, 255, 0), 2)
    # cv2.drawContours(im_gray, contours1, -1, (0, 255, 0), 2)
    # cv2.imshow("Image", im_gray)
    # cv2.waitKey(0)
    # loop over the contours
    for c in contours1:
        # draw the contour and show it
        # cv2.drawContours(im2, contours1, 4, (255,0,0), 2)
        cv2.drawContours(im_gray, [c], -1, (0, 255, 0), 2)
        cv2.imshow("Image", im_gray)
        cv2.waitKey(0)

if __name__ == '__main__':

    parent_dir = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images"

    imagePath1 = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images\118_case769_front_Top_Bottom_0.jpg"
    imagePath1_out = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images\118_case769_front_Top_Bottom_0_out.jpg"
    imagePath2 = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Images\118_case769_front_Top_Bottom_-1.jpg"

    #image3 = get_diff(image1, image2)

    parentDir = parent_dir #os.path.split(imPath1)[0]
    # load the two input images
    # imageA = cv2.imread(image1)
    # imageB = cv2.imread(image2)
    #
    get_diff(imagePath1, imagePath2)
    # Image extractor from Python Imaging Library
    #get_diff1(imagePath1, imagePath2)
    #get_diff2(imagePath1, imagePath2)
    #get_outline(imagePath1)
    # read_image2(imagePath1, imagePath1_out)
