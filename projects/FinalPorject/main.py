from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imutils
import pytesseract

from gaussian_filtering import gaussian_filtering
from histogram_equalization import histogram_equalization
from my_sobel import my_sobel
from non_max_suppression import non_max_suppression
from morphology import test
from alignment import alignImages

img = mpimg.imread('image_0009.jpeg')  # dataset

# histogram_equalization(img)

# gaussian_filtering(img)

#test(img)

# alignImages(img, img2)

img = cv2.GaussianBlur(img, (5, 5), 2)
img2 = histogram_equalization(img)
image = img2

edged = cv2.Canny(img2, 30, 200)
cv2.imshow("edged image", edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("contours", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 contours", image2)
cv2.waitKey(0)

i = 99
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        new_img = image[y:y + h, x:x + w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
