# Author: Brycen

from Pyramids.laplacian_pyr import gaussian_pyr, laplacian_pyr, laplacian_reconstruct
import cv2
import matplotlib.pyplot as plt
import numpy as np

def blend(filename, mask, levels):
    # mask = cv2.imread('./blend/mask/' + filename) / 255.0
    imageA = cv2.imread('./images/blend/imageA/' + filename) / 255.0
    imageB = cv2.imread('./images/blend/imageB/' + filename) / 255.0

    mask_pyr = gaussian_pyr(mask, levels)
    imageA_pyr = laplacian_pyr(imageA, levels)
    imageB_pyr = laplacian_pyr(imageB, levels)

    new_pyr = (imageA_pyr * mask_pyr) + (imageB_pyr * (1 - mask_pyr))
    reconstructed = laplacian_reconstruct(new_pyr, levels)
    return np.flip(reconstructed, axis=2) # convert opencv BGR to RGB

