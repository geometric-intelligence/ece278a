'''-------------------------------------------
Created By:
Roger Lin

Binarize Image Using OTSU
--------------------------------------------'''
from skimage.filters.thresholding import threshold_otsu


def img_bin(img_gray):
    threshold = threshold_otsu(img_gray)
    binary_img = img_gray > threshold

    return binary_img

