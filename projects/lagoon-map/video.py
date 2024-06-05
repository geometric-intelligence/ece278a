import cv2
import numpy as np
import os

# def crop(image):
#     y_nonzero, x_nonzero, _ = np.nonzero(image)
#     return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

width = 256

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=lambda filename: int(filename.split('.')[0])):
        img = cv2.imread(os.path.join(folder,filename))
        if img is None: break
        
        # img = crop(img)
        img = img[int((img.shape[0]-width)/2):int((img.shape[0]+width)/2), int((img.shape[1]-width)/2):int((img.shape[1]+width)/2)]
        images.append(img)

    print(f'Loaded {len(images)} images')
    return images

# Load the images
images = load_images_from_folder('radar_image')

size = 256, 256
fps = 10
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
for image in images:
    video.write(image)
video.release()
