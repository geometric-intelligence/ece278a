import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2


def gaussian_filtering(img):
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 1)

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img_gaussian)
    plt.title('Smoothed')
    plt.xticks([])
    plt.yticks([])
    plt.show()
