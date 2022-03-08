# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:37:29 2020

@author: aniket wattamwar
"""

from re import M, S
from turtle import color, width
import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import skimage
import matplotlib.pyplot as plt
import random
import math

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Pairwise Alignment Theory', 'Pairwise Alignment Demo', 'Limitation & Robustness', 'RANSAC Demo')
    )

    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Pairwise Alignment Theory':
        theory()
    if selected_box == 'Pairwise Alignment Demo':
        pademo()
    if selected_box == 'Limitation & Robustness':
        robust()
    if selected_box== 'RANSAC Demo':
        rsdemo()

def welcome():
    st.title('Image Alignment')
    st.image('DMV.png',use_column_width=True)
    return




def theory():
    st.title('Pairwise Alignment')
    st.subheader('General Method of Pairwise Alinment')
    st.text('1) Find N matched keypoints in 2 images')
    st.latex(r'''
        \left\{x_i, x_i'\right\}_{i=1}^N
    ''')
    st.text('2) Find the transformation function')
    st.latex(r'''
        \hat x' = f(x; p)
    ''')

    st.subheader('Using Least Square(LS) to find the Parameter p')
    st.latex(r'''
        min_p E_{LS}(p) = \sum_{i=1}^N ||r_i||^2 = \sum_{i=1}^N ||f(x_i; p) - x_i'||^2
    ''')

    st.subheader('Linear least squares')
    st.text('Many transformations are linear, and they will have a linear relationship between:')
    st.text('The amount of motion ')
    st.latex('''\Delta \hat x = \hat x' - x''')
    st.text('and The Jacobian of x')
    st.latex('''\Delta \hat x = \hat x' - x = J(x) p''')
    st.text('Using this linear relationship:')
    st.image('lmath.png',use_column_width=True)
    st.text('Finally we can find the linear regression in p, by solving the equation:')
    st.latex('''Ap = b''')

    st.subheader('Nonlinear least squares')
    st.text('Other tranformations are not linear, and they do not have the linear realationship\nbetween the amount of motion and the Jacobian of x')
    st.text('In these transformations the Jacobian of x depends on p')
    st.latex('''E_{NLS}(\Delta p) =\sum_{i=1}^N ||f(x_i; p + \Delta p) - \Delta x_i||^2''')
    st.text('Keep updating Delta p until convergence')
    st.image('nmath.png',use_column_width=True)
    st.text('Finally we can find Delta p by solving:')
    st.latex('(A + \lambda diag(A)) \Delta p = b')
    return


def load_image(filename):
    image = cv2.imread(filename)
    return image

def pademo():
    st.title('Pairwise Alignment Demo')
    imref=load_image('dollar.jpg')
    imrefrgb=cv2.cvtColor(imref,cv2.COLOR_BGR2RGB)
    imrefgray=cv2.cvtColor(imref,cv2.COLOR_BGR2GRAY)
    st.text('The reference image')
    st.image(imrefrgb)
    selected_box = st.selectbox(
    'Choose the image',
    ('Dollar1.jpg', 'Dollar2.jpg', 'Dollar3.jpg')
    )
    im = load_image(selected_box)
    imrgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    st.text('The image to be aligned')
    st.image(imrgb)

    st.subheader('Make some transformation')
    degree = st.slider('Rotation',min_value = 0,max_value = 360)
    width = st.slider('Width Scale',min_value = 0.1,max_value = 2.0)
    height = st.slider('Hight Scale',min_value = 0.1,max_value = 2.0)
    
    w = int(im.shape[1] * width)
    h = int(im.shape[0] * height)
    dim = (w, h)
    ims = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    imsr = rotate_image(ims,degree)

    imsrrgb=cv2.cvtColor(imsr,cv2.COLOR_BGR2RGB)
    st.image(imsrrgb)

    imReg, h = alignImages(imsr, imref)

    st.image(cv2.cvtColor(imReg,cv2.COLOR_BGR2RGB))

    return

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h


def robust():
    st.title('Robustness')
    st.subheader('Robustness via Weighted LS')
    st.text('Associate a scalar variance estimate and use the weighted least squares:')
    st.latex('''min_p E_{LS}(p)  =\sum_{i=1}^N \sigma^{-2}_i||r_i||^2''')
    st.subheader('Robustness via Keypoint Pre-Selection')
    st.text('Get rid of the outlier')
    st.image('04_outliers.png')
    st.header('Least Median Square (LMS)')
    st.text('Select at random a subset k of N matches')
    st.text('Compute geometric transformation parameter p_hat of transformations')
    st.text('Compute the residuals r for all keypoints')
    st.text('Compute the median med is the median of current residuals')
    st.text('Keep sample k with smallest median residual, and corresponding p_hat')
    st.subheader('RANdom SAmple Consensus RANSAC')
    st.text('Select at random a subset of k of the N matches')
    st.text('Compute geometric transformation parameter p_hat of transformations')
    st.text('Compute the residuals r')
    st.text('Count the number of "inliers"')
    st.text('Repeat selection process S times')
    st.text('Keep sample of k keypoints with largest number of inliers')
    st.text('Recompute on all inliers')
    return
         
def rsdemo():
    st.title('RANSAC Demo')
    SIZE = st.slider('Data Size',min_value = 50,max_value = 500)
    X = np.linspace(0, 10, SIZE)
    Y = 3 * X + 10
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title("RANSAC")

    # add noise
    random_x = []
    random_y = []   
    for i in range(SIZE):
        random_x.append(X[i] + random.uniform(-0.5, 0.5)) 
        random_y.append(Y[i] + random.uniform(-0.5, 0.5)) 

    for i in range(SIZE):
        random_x.append(random.uniform(0,10))
        random_y.append(random.uniform(10,40))
    
    RANDOM_X = np.array(random_x)
    RANDOM_Y = np.array(random_y)
    
    ax1.scatter(RANDOM_X, RANDOM_Y)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    iters = st.slider('Iterations (S)',min_value = 1,max_value = 100)

    sigma = st.slider('Tolerance (epsilon)',min_value=0.01, max_value=1.00)

    st.text('Here, for simplicity I pick k=2, which means I use 2 points for regression every time')

    best_a = 0
    best_b = 0
    pretotal = 0
    bx_1=0
    bx_2=0
    by_1=0
    by_2=0

    P = 0.99
    for i in range(iters):
        sample_index = random.sample(range(SIZE * 2),2)
        x_1 = RANDOM_X[sample_index[0]]
        x_2 = RANDOM_X[sample_index[1]]
        y_1 = RANDOM_Y[sample_index[0]]
        y_2 = RANDOM_Y[sample_index[1]]

        a = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - a * x_1


        total_inlier = 0
        for index in range(SIZE * 2):
            y_estimate = a * RANDOM_X[index] + b
            if abs(y_estimate - RANDOM_Y[index]) < sigma:
                total_inlier = total_inlier + 1

        if total_inlier > pretotal:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
            pretotal = total_inlier
            best_a = a
            best_b = b
            bx_1=x_1
            bx_2=x_2
            by_1=y_1
            by_2=y_2
            st.text(best_a)

    Y = best_a * RANDOM_X + best_b
    ax1.plot(RANDOM_X, Y)
    text = "best_a = " + str(best_a) + "\nbest_b = " + str(best_b)
    plt.text(5,10, text,
            fontdict={'size': 8, 'color': 'r'})
    plt.show()

    st.pyplot(fig)
    return
    

    
    
    
if __name__ == "__main__":
    main()
