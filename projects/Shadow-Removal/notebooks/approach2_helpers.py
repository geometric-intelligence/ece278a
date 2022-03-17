"""Helper functions for Approach #2.

Defines helper functions that are used to perform Approach #2
"""

# import necessary packages
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from skimage.filters import threshold_otsu

def read_img(filename, isRGB=False):
    
  img = mpimg.imread(filename)
  if not isRGB: img = color.rgb2gray(img) # convert to grayscale
  plt.figure(figsize=(12,5))
  plt.axis('off')
  plt.title('Original Image')
  plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
  return img


def get_LocalBG(img, kernel_size=5):
  
  L = None
  if len(img.shape) == 3:
    temp_L = []
    for i in range(3):
      Imax = ndimage.maximum_filter(img[:,:,i], size=kernel_size) # apply max filter
      temp_L.append(Imax)
      temp_L[i][temp_L[i] == 0] = 0.000001 # to prevent NaN due to divide by 0
    L = np.stack((temp_L), axis=2)
  else:
    Imax = ndimage.maximum_filter(img, size=kernel_size) # apply max filter
    L = Imax
    L[L == 0] = 0.000001 # to prevent NaN due to divide by 0

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
  ax1.axis('off')
  ax1.set_title('Original Image')
  ax1.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax2.axis('off')
  ax2.set_title('Local Background Color Image')
  ax2.imshow(L, cmap=plt.cm.gray, vmin=0, vmax=1)
  return L

def show_ShadowMap(L):

  shadow_map = None
  if len(L.shape) == 3:
    shadow_maps = []
    for i in range(3):
      thresh = threshold_otsu(L[:,:,i]) # binarize image with thresholding
      temp_shadow_map = L[:,:,i] < thresh
      temp_shadow_map = temp_shadow_map * 1
      shadow_maps.append(temp_shadow_map)
    shadow_map = shadow_maps[0] | shadow_maps[1] | shadow_maps[2]
  else:
    thresh = threshold_otsu(L) # binarize image with thresholding
    shadow_map = L < thresh

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
  ax1.axis('off')
  ax1.set_title('Local Background Color Image')
  ax1.imshow(L, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax2.axis('off')
  ax2.set_title('Shadow Map')
  ax2.imshow(shadow_map, cmap=plt.cm.gray, vmin=0, vmax=1)

def get_GlobalBG(L):

  unshadow_bg = None
  G = None
  if len(L.shape) == 3:
    unshadow_bgs = []
    for i in range(3):
      thresh = threshold_otsu(L[:,:,i]) # binarize image with thresholding
      unshadowed_area = L[:,:,i] > thresh
      temp_unshadow_bg = L[:,:,i] * unshadowed_area # mask the shadowed areas
      unshadow_bgs.append(temp_unshadow_bg)
      unshadow_bg = np.stack((unshadow_bgs), axis=2)

    for i in range(3):
      unshadow_bgs[i] = unshadow_bgs[i][unshadow_bgs[i] != 0]
      unshadow_bgs[i] = np.mean(unshadow_bgs[i]) # average pixel color
      unshadow_bgs[i] = unshadow_bgs[i] * np.ones(L[:,:,i].shape)
      unshadow_bgs[i] = unshadow_bgs[i].astype(int)
    G = np.stack((unshadow_bgs), axis=2)

  else:
    thresh = threshold_otsu(L) # binarize image with thresholding
    unshadowed_area = L > thresh
    unshadow_bg = L * unshadowed_area # mask the shadowed areas

    temp_unshadow_bg = unshadow_bg[unshadow_bg != 0]
    temp_unshadow_bg = np.mean(temp_unshadow_bg) # average pixel color
    G = temp_unshadow_bg * np.ones(L.shape)

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
  ax1.axis('off')
  ax1.set_title('Shadow-Free Local Background Image')
  ax1.imshow(unshadow_bg, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax2.axis('off')
  ax2.set_title('Global Background Color')
  ax2.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)
  return G

def get_FinalImg(img, L, G):

  r = G/L # find shadow scale
  final = r * img # relight shadow regions
  if len(img.shape) == 3: final = final.astype(int)

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
  ax1.axis('off')
  ax1.set_title('Original Image (with shadow)')
  ax1.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax2.axis('off')
  ax2.set_title('Final Image (without shadow)')
  ax2.imshow(final, cmap=plt.cm.gray, vmin=0, vmax=1)
  return final

def fineTune(img, L, G, final):

  thresh = threshold_otsu(L) # binarize image with thresholding
  shadow_map = L < thresh
  shadow_bg = L * shadow_map # mask the unshadowed areas
  shadow_bg = shadow_bg[shadow_bg != 0]
  shadow_bg = np.mean(shadow_bg) # average pixel color
  shadow_bg = shadow_bg * np.ones(L.shape)

  tau = (G/shadow_bg) * 0.5 # find tone scale
  tuned = final * tau # apply tone scale
  if len(img.shape) == 3: tuned = tuned.astype(int)

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,8))
  ax1.axis('off')
  ax1.set_title('Original Image')
  ax1.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax2.axis('off')
  ax2.set_title('Unshadowed Image')
  ax2.imshow(final, cmap=plt.cm.gray, vmin=0, vmax=1)
  ax3.axis('off')
  ax3.set_title('Tone Fine-Tuned Image')
  ax3.imshow(tuned, cmap=plt.cm.gray, vmin=0, vmax=1)