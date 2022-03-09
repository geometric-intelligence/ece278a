# Author: Brycen

import streamlit as st
import cv2
from Pyramids.laplacian_pyr import gaussian_pyr, laplacian_pyr, laplacian_display
from Pyramids.image_blend import blend
import numpy as np

def display_pyramids():
    st.header('Image Resampling')
    st.write('Image resampling is an important and common procedure in image processing that allows \
        images to be represented at different scales. Image upsampling (interpolation) is a technique used to \
        approximate higher resolution images while image downscaling (decimation) is used to \
        create lower resolution images while avoiding aliasing')

    st.subheader('Interpolation')
    st.write('The main idea of interpolation is to approximate a continuous representation of the \
        image using the existing samples, and then use that continuous approximation to resample at a \
        higher rate. There are several different types of interpolation kernels that can be used for \
        this purpose depending on the quality and computational efficiency suited for the application. \
        The approximation and sampling steps of image interpolation are typically combined and implemented in the \
        form of a modified convolution:')
    st.latex(r'''g\left(i, j\right) = \sum_{k,l} f(k, l)h(i - rk, j - rl)''')
    st.write('Where r is the scalar upsampling rate, f() is the original image, and h() is the interpolation kernel')
    st.image('./images/figures/interpolation.png', caption='Interpolation Techniques')
    st.write('Common Interpolation Kernels')
    st.write('Bilinear: Bilinear interpolation is generates a peiceswise linear representation of the source image \
        to resample. It is a computationally efficient algorithm that only requires a 2x2 convolution kernel. \
        The main drawback of this method is that the approximation is not continuously differentiable which can result \
        in unappealing creasing in the rescaled image.')
    st.write('Bicubic: Bicubic interpolation is the result of fitting cubic splines to the image data and results \
        in a smoother, more visually appealing result when compared with bilinear interpolation. \
        The main drawback of this technique is that the interpolation convolution requires a 4x4 kernel \
        making it more computationally expensive than Bilinear interpolation')
    st.image('./images/figures/interpolation_compare.png', caption='Visual comparison of interpolation kernels on a sample distribution')
    st.subheader('Decimation')
    st.write('Decimation is the process of reducing a resolution by first blurring the image to avoid aliasing and then only keeping \
        every rth sample where integer r is the decimation rate. In the case that r is not an integer, the image can fist be interpolated \
        by integer factor L, then decimated by integer factor M such that r = L/M. Decimation can also be implemented in the form of a \
        modified convolution with form:')
    st.latex(r'''g\left(i, j\right) = \frac{1}{r}\sum_{k,l} f(k, l)h(i - k / r, j - l / r)''')
    st.write('Where r is the decimation rate, f() is the original image, and h() is the smoothing kernel')
    st.write('Much like interpolation, there are several kernel options to choose from, each with different frequency cutoff properties.')
    st.header('Pyramids')
    st.write('Image pyramids are multiscale respresentations of images that are very useful for computer vision tasks \
        such as coarse-to-fine image search algorithms and multiscale pattern recognition/feature tracking.')
    st.subheader('Gaussian Pyramid')
    st.write('A Gaussian pyramid is constructed by successively decimating an image by a factor of 2 until \
        the desired number of levels is reached. Despite what the name suggests, the decimation blurring kernel used is typically in the \
        form of a binomial distribution. The name orginates from the fact that repeated convolutions of \
        the binomial kernel converges to a gaussian rather than the use of a gaussian kernel.')
    select_image = st.selectbox("Image", ('tiger', 'mountains', 'woman'))
    pyr_img = cv2.imread('./images/{}.png'.format(select_image))
    pyr_img = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2RGB) / 255.0
    gauss_levels = st.slider('Pyramid Levels', 1, 6, 3, key=1)
    st.image(gaussian_pyr(pyr_img, gauss_levels-1))
    st.subheader('Laplacian Pyramid')
    st.write('A Laplacian pyramid can be computed from a corresponding Gaussian pyramid by interpolating each level other than the original \
        by a factor of 2, then taking the difference between the gaussian pyramid and the interpolated results. The result of the difference \
        between the low pass Gaussian images and \"lower\" pass interpolated images is a bandpass representation of the original image \
        at different frequency bands. The name Laplacian pyramid comes from the idea that the levels of the pyramid are approximately the \
        same as convolving the original images with a Laplacian of a Gaussian kernel')
    laplace_levels = st.slider('Pyramid Levels', 1, 6, 3, key=2)
    st.image(np.clip(laplacian_display(laplacian_pyr(pyr_img, laplace_levels), laplace_levels), 0, 1), caption='Note: \
        Though it is sometimes not pictured in the Laplacian Pyramid, the smallest level of the Gaussian pyramid is needed to be able to \
        reconstruct the original image')

    st.header('Application: Image Blending')
    st.write('One interesting of Laplacian and Gaussian Pyramids is image blending. This Blending techiniques works by blurring together \
        low frequency components of two images while retaining the high frequency features of both. It is a simples procedure that utilizes \
        Laplacian Pyramids of the images and a Gaussian pyramid of the mask image.')
    st.latex(r'''l_k = l_k^A*m_k + l_k^B *(1 - m_k)''')
    mask_extent = st.slider('Extent of Mask', 0, 100, 50)
    mask_index = int(mask_extent / 100 * 512)
    mask = np.zeros((512, 512, 3))
    mask[:, 0:mask_index] = 1
    st.image(np.clip(blend('fruitBlend.png', mask, 6), 0, 1), width=700)