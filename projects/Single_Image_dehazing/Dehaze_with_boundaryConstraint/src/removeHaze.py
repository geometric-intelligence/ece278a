import cv2
import numpy as np
import copy

def removeHaze(hazy_img, Transmission, airlight, delta):

    epsilon = 0.0001
    Transmission = pow(np.maximum(abs(Transmission), epsilon), delta)

    HazeCorrectedImage = copy.deepcopy(hazy_img)
    if(len(hazy_img.shape) == 3):
        for ch in range(len(hazy_img.shape)):
            temp = ((hazy_img[:, :, ch].astype(float) - airlight[ch]) / Transmission) + airlight[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage[:, :, ch] = temp
    else:
        temp = ((hazy_img.astype(float) - airlight[0]) / Transmission) + airlight[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        HazeCorrectedImage = temp
    return(HazeCorrectedImage)