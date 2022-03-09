'''-------------------------------------------
 UCSB ECE 278A - Image Processing
 Webapp - Image Segmentation

Created By
Tainan Song
Roger Lin

This file contains streamlit section structures:
    intro()
    threshold()
    region()
    cluster()
--------------------------------------------'''
import streamlit as st
from lib.imgproc import *
from skimage import data


def intro():
    st.text('Image segementation is the process of dividing an image into different regions \n' +
            'based on the characteristics of pixels to identify objects or boundaries.')
    st.header('Types of Segmentation')
    st.text('Semantic Segmentation')
    st.image('source/eg_sementic_seg.png')
    st.text('Foreground/Background Segmentation')
    st.image('source/eg_foreground_seg.jpeg')
    st.header('Applications')
    st.text('- Content based image retrieval \n' +
            '- Medical imaging  \n' +
            '- Object detection \n' +
            '- 3D reconstruction \n' +
            '- Object tracking, object recognition\n' +
            '- Object-based measurements \n' +
            '- Video surveillance\n')
    st.text('Photo Editing')
    st.image('source/eg_por_seg.jpeg')
    st.text('Object Detection')
    st.image('source/eg_translate_seg.jpeg')
    st.header('Algorithms')
    st.text('- Thresholding \n' +
            '- Region-based methods \n' +
            '- Clustering \n' +
            '- Graph-based methods \n' +
            '- Shape based methods\n' +
            '- Energy minimization methods\n' +
            '- Machine learning and deep learning methods')


def threshold():
    st.subheader('Binarization')
    st.text("Binarization simply applies a global threshold to map a grayscale image into a \n" +
            "a black and white image to seperate out darker and lighter objects.")
    st.latex("J(x, y) =  0 \quad if \  I(x, y) < T ")
    st.latex("J(x, y) =  1 \quad if \ otherwise")
    def_img = skimage.io.imread('source/2019-11-30 11.51.39.jpg')
    img = load_img(def_img, 1)
    binarization(img)

    st.subheader('Otsu Thresholding')
    st.text("Otsu further improves the threshold value using iterations to maximize the variance \n" +
            "between two classes.")
    st.latex("\sigma_b^2 = P_1 (\mu_1 - \mu)^2 + P_2 (\mu_2 - \mu)^2 = P_1 P_2 (\mu_1 - \mu_2)^2")
    img = load_img(def_img, 2)
    otsu(img)


def region():
    st.subheader('Region Growing')
    st.text("Start with a seed pixel and add similar adjacent pixels recursively")
    st.image('source/eg_reg_grow.png')
    def_img = skimage.io.imread('source/2019-03-30 16.21.51.jpg')
    img = load_img(def_img, 3)
    region_grow(img)

    st.subheader('Region Splitting and Merging')
    st.text("1. Start with whole image and split recursively until a homogeneity condition is satisfied \n" +
            "2. Start with small regions and merge similar regions recursively")
    st.text("- This method avoids oversegmentation")
    st.image('source/eg_reg_split.pbm')
    def_img = skimage.io.imread('source/2016-08-14 16.42.52-1.jpg')
    img = load_img(def_img, 4)
    region_splitting_merging(img)

    st.subheader('Over-segmentation')
    st.text("If the image is over-segmentated, we would not be able to separate the zebra out.")
    st.image('source/eg_zebra.jpeg')


def cluster():
    st.text('Clustering organizes data into groups that with high/low intra-class similarity. ')
    st.subheader('Mean-Shift')
    st.latex("m(x) = \\frac{\sum_{x_i \in N(x)} K(x_i - x)x_i}{\sum_{x_i \in N(x)} K(x_i - x)}")
    st.image('source/eg_meanshift.png')
    st.image('source/eg_meanshift1.png')
    def_img = skimage.io.imread('source/cars.jpeg')
    img = load_img(def_img, 5)
    quick_shift(img)


