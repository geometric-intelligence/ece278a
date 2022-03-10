from image_functions import guidedFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2

imgcolor = cv2.imread('./cat.png')
imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2RGB)
imggray = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2GRAY) / 255.0
imgcolor = imgcolor / 255.0
work = np.zeros((imggray.shape[0], imggray.shape[1], 1))
work[:, :, 0] = imggray

blur = cv2.GaussianBlur(work, (21, 21), 20)

guided = guidedFilter(imgcolor, blur, 5)
guided_opencv = (guided * 255).astype(np.uint8)

#cv2.imwrite('cat_smooth_with_edges.png', guided_opencv)

plt.figure()
plt.imshow(blur, cmap='gray')
plt.figure()
plt.imshow(guided, cmap='gray')
plt.show()


