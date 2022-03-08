'''-------------------------------------------
 UCSB ECE 278A - Image Processing
 Webapp - Image Segmentation

Created By
Tainan Song
Roger Lin

This file contains image processing functions:
    load_img(default_image, uploder_key)
    convert_to_grey(img_file)
     binarization(img_file)
     otsu(img_file)
     region_grow(img_file)
     region_splitting_merging(img_file)
     quick_shift(img_file)
     _weight_mean_color(graph, src, dst, n)
     merge_mean_color(graph, src, dst)
--------------------------------------------'''
import numpy as np
import skimage.io
from scipy import ndimage
import streamlit as st
import matplotlib.pyplot as plt
from lib.webapp import *
from skimage import color
from skimage.segmentation import *
from skimage.filters import *
from skimage.future import graph
from skimage.util import img_as_float


#
# Load image from user as np.array, otherwise use default image provided
#
def load_img(default_image, uploder_key):
    img = default_image
    uploaded_imag = st.file_uploader('Upload Image in JPEG', type='jpg', key=uploder_key)
    if uploaded_imag is not None:
        img = skimage.io.imread(uploaded_imag)
    return img


#
# Convert image to greyscale if is not, rescale intensity to 0-255
#
def convert_to_grey(img_file):
    if len(img_file.shape) == 3:
        img = 255*color.rgb2gray(img_file)
    else:
        img = img_file
    return img

#
# Apply global threshold set by slider for binarization
#
def binarization(img_file):
    # Image Binarization with Thresholding
    img_gray = convert_to_grey(img_file)
    threshold = st.slider('Change Global Threshold Value', min_value=0, max_value=255, value=100)
    binary = img_gray > threshold

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(img_file,  cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].axvline(threshold, color='r')
    ax[1].hist(img_gray.ravel(), bins=256)
    ax[1].set_title('Grayscale Histogram')
    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Processed Image')
    ax[2].axis('off')
    st.pyplot(fig)


#
# Apply otsu thresholding for binarization
#
def otsu(img_file):
    img_gray = convert_to_grey(img_file)
    threshold = threshold_otsu(img_gray)
    st.write('Threshold = ' + str(threshold))
    binary = img_gray > threshold

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(img_file,  cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].axvline(threshold, color='r')
    ax[1].hist(img_gray.ravel(), bins=256)
    ax[1].set_title('Grayscale Histogram')
    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Processed Image')
    ax[2].axis('off')
    st.pyplot(fig)


#
# Apply random_walker to demonmstrate region growing method
#
def region_grow(img_file):
    img_gray = convert_to_grey(img_file)
    binary_marker_thres = st.slider('Create two markers separated by intensity value of :', min_value=0, max_value=255, value=100)
    markers = np.zeros(img_gray.shape, dtype=np.uint)
    markers[img_gray <= binary_marker_thres] = 1
    markers[img_gray > binary_marker_thres] = 2
    labels = random_walker(img_gray, markers)

    fig, axes = plt.subplots(ncols=2, figsize=(18, 6))
    axes[0].imshow(img_file,  cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(labels, cmap='gray')
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    st.pyplot(fig)


#
# Apply slic and graph to demonmstrate region splitting and merging
#
def region_splitting_merging(img_file):
    # Region Splitting and Merging
    img = img_file
    st.text("Balances color proximity and space proximity. Higher values give more weight to space proximity")
    compact = st.slider('Compactness:', min_value=1, max_value=100, value=50)
    st.text("The approximate number of labels in the segmented output image")
    seg_number = st.slider('Number of Segments:', min_value=1, max_value=500, value=50)
    labels = slic(img, compactness=compact, n_segments=seg_number, start_label=1)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = mark_boundaries(out, labels2, (0, 0, 0))
    fig, axes = plt.subplots(ncols=2, figsize=(18, 6))
    axes[0].imshow(img)
    axes[1].imshow(out)
    st.pyplot(fig)


#
# Apply quick shift to demonmstrate mean-shift
#
def quick_shift(img_file):
    img = img_as_float(img_file[::2, ::2])
    kernal = st.slider('Kernal Size:', min_value=1, max_value=100, value=20)
    dist = st.slider('Max Distance:', min_value=1, max_value=100, value=20)
    segments_quick = quickshift(img, kernel_size=kernal, max_dist=dist)
    st.write('Number if segments: ' + str(len(np.unique(segments_quick))))

    fig, axes = plt.subplots(ncols=2, figsize=(18, 6))
    axes[0].imshow(img_file, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(mark_boundaries(img, segments_quick))
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    st.pyplot(fig)

#
# Functions Created By Prof. Nina Miolane
# https://github.com/MarugoBazu/ece278a/blob/main/lectures/03_feature_detection_matching.ipynb
#
def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])