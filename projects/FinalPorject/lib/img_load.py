'''-------------------------------------------
Created By:
Roger Lin

Load images from file as numpy array
--------------------------------------------'''
from skimage import io
from skimage import color


def img_load(image_path, **kwargs):
    gray = kwargs.get('gray', False)
    if image_path is not None:
        img = io.imread(image_path)
        if gray:
            img = color.rgb2gray(img)
    return img


