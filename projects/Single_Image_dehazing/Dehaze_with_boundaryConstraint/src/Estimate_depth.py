import cv2
import numpy as np

def Estimate_lowerbound(hazy_img, airlight, C0, C1, window_size):
    if(len(hazy_img.shape) == 3):
        t_x_blue = np.maximum((airlight[0] - hazy_img[:, :, 0].astype(np.float)) / (airlight[0] - C0),
                         (hazy_img[:, :, 0].astype(np.float) - airlight[0]) / (C1 - airlight[0]))
        t_x_green = np.maximum((airlight[1] - hazy_img[:, :, 1].astype(np.float)) / (airlight[1] - C0),
                         (hazy_img[:, :, 1].astype(np.float) - airlight[1]) / (C1 - airlight[1]))
        t_x_red = np.maximum((airlight[2] - hazy_img[:, :, 2].astype(np.float)) / (airlight[2] - C0),
                         (hazy_img[:, :, 2].astype(np.float) - airlight[2]) / (C1 - airlight[2]))

        MaxVal = np.maximum(t_x_blue, t_x_green, t_x_red)
        t_b = np.minimum(MaxVal, 1)
    else:
        t_b = np.maximum((airlight[0] - hazy_img.astype(np.float)) / (airlight[0] - C0),
                         (hazy_img.astype(np.float) - airlight[0]) / (C1 - airlight[0]))
        t_b = np.minimum(t_b, 1)

    kernel = np.ones((window_size, window_size), np.float)
    t_b = cv2.morphologyEx(t_b, cv2.MORPH_CLOSE, kernel=kernel)
    return(t_b)

