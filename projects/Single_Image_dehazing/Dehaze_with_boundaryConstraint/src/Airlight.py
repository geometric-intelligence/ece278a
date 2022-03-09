import numpy as np
import cv2

def Airlight(hazy_Img, windowSize):
    airlight = []
    if(len(hazy_Img.shape) == 3):
        for ch in range(len(hazy_Img.shape)):
            kernel = np.ones((windowSize, windowSize), np.uint8)
            eroded_img = cv2.erode(hazy_Img[:, :, ch], kernel)
            airlight.append(int(eroded_img.max()))
    else:
        kernel = np.ones((windowSize, windowSize), np.uint8)
        eroded_img = cv2.erode(hazy_Img, kernel)
        airlight.append(int(eroded_img.max()))
    return(airlight)
