from image_functions import guidedFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2

dog = cv2.imread('./dog.png')
dog_scaled = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB) / 255.0
mask = cv2.imread('./dog_mask.png', 0) / 255.0
mask = mask

guided = guidedFilter(dog_scaled, mask, 25)
#guided = np.where(guided < 0.25, guided, 255)
guided = (np.clip(guided, 0, 1) * 255).astype(np.uint8)

dog_alpha = cv2.cvtColor(dog, cv2.COLOR_BGR2BGRA)
dog_alpha[:, :, 3] = 255 - guided

cv2.imwrite('refined_dog_mask2.png', guided)

plt.figure()
plt.imshow(guided, cmap='gray')
plt.show()
