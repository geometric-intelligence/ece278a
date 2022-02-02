
"""
Created on Tue Oct 27 20:37:29 2020
@author: aniket wattamwar
"""

import streamlit as st
from PIL import Image
import cv2
import numpy as np

from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
# import numpy as np
import requests
from io import BytesIO


def main():
    selected_box = st.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'Homography')
    )

    if selected_box == 'Welcome':
        welcome()

    if selected_box == 'Homography':
        Homography()


def welcome():
    st.title('Hi there, welcome to our web app')

    st.subheader('ECE 278A Digital Image Processing @UCSB')
    st.subheader('Team: Sean MacKenzie, Rami Dabit and Peter Li.')
    st.subheader('Feel free to interact with our web app.')

    # st.image('hackershrine.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image


def Homography():
    st.header("Image-Formation: Homography")

    # ========================================================
    # my own start

    url = 'https://images.unsplash.com/photo-1613048998835-efa6e3e3dc1b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80'

    response = requests.get(url)
    imgfile = Image.open(BytesIO(response.content))
    img = np.array(imgfile)

    if st.button('Original Image'):
        # [Remind] use st.image to plot
        st.image(img, use_column_width=True)

    my_phi = st.slider('Change angle to decide camera position', min_value=-35, max_value=35, value=0)
    my_k = st.slider('Change Value to zoon in or zoom out', min_value=-0.2, max_value=1.0, value=0.2)

    # Setting Parameter
    # phi = 25 # [-70~70]
    # unit degree
    phi = my_phi
    scale_factor = 1  # (this is optional)

    # k = 0.7 # need to be positive-value
    k = my_k
    b = 0.5  # fix

    increment = ((k + b) / b) * np.tan((phi / 180) * 3.14)
    l_side = np.sqrt(((k + b) / b) ** 2 + (k + b) ** 2) + increment
    r_side = np.sqrt(((k + b) / b) ** 2 + (k + b) ** 2) - increment

    origin_len = img.shape[0] / 2
    origin_wid = img.shape[1] / 2

    transform_center = [origin_wid, origin_len]

    transform_l = (np.sqrt(1 + b ** 2) / l_side) * origin_len
    transform_r = (np.sqrt(1 + b ** 2) / r_side) * origin_len
    transform_wid = origin_wid * (b / (k + b))

    # source coordinates
    src_i = np.array([0, 0,
                      0, img.shape[0],
                      img.shape[1], img.shape[0],
                      img.shape[1], 0, ]).reshape((4, 2))

    # destination coordinates
    dst_i = np.array(
        [transform_center[0] - transform_wid * scale_factor, transform_center[1] - transform_l * scale_factor,
         transform_center[0] - transform_wid * scale_factor, transform_center[1] + transform_l * scale_factor,
         transform_center[0] + transform_wid * scale_factor, transform_center[1] + transform_r * scale_factor,
         transform_center[0] + transform_wid * scale_factor,
         transform_center[1] - transform_r * scale_factor, ]).reshape((4, 2))

    # using skimage’s transform module where ‘projective’ is our desired parameter
    tform = transform.estimate_transform('projective', src_i, dst_i)
    tf_img = transform.warp(img, tform.inverse)

    # plotting the original image
    plt.imshow(img)

    # plotting the transformed image
    fig, ax = plt.subplots()
    ax.imshow(tf_img)
    _ = ax.set_title('projective transformation')
    plt.plot(transform_center[0], transform_center[1], 'x')
    plt.show()

    # streamlit explanation
    if my_phi > 0:
        direction_string = "rotate from original position to right at " + str(my_phi)+ " degrees."
    elif my_phi < 0:
        direction_string = "rotate from original position to left at " + str(-my_phi)+ " degrees."
    else:
        direction_string = "is at original position ."

    if my_k > 0:
        distance_string = " Zoom out."
    elif my_k < 0:
        distance_string = " Zoom in."
    else:
        distance_string = " No zoom in/out." 

    string_camera_posi = " Camera {} ".format(direction_string)
    string_zoomInOut  = "  {} ".format(distance_string)

    st.caption(body=string_camera_posi)
    st.caption(body=string_zoomInOut)



    # [Remind] use st.image to plot
    st.image(tf_img, use_column_width=True)

    # my own end
    # ========================================================


if __name__ == "__main__":
    main()