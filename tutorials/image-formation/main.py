# notes

# imports
import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np

from skimage.io import imread, imshow
from skimage import transform
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import requests
from io import BytesIO


def main():
    selected_box = st.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'Pinhole Camera Model', 'Paraxial Camera Model', 'Homography',
         'Camera Intrinsics', 'Camera Extrinsics')
    )

    if selected_box == 'Welcome':
        welcome()

    if selected_box == 'Pinhole Camera Model':
        pass

    if selected_box == 'Paraxial Camera Model':
        paraxial_camera_model()

    if selected_box == 'Homography':
        homography()

    if selected_box == "Camera Intrinsics":
        #camera_intrinsics()
        pass


def welcome():
    st.title('Hi there, welcome to our web app')

    st.subheader('ECE 278A Digital Image Processing @UCSB')
    st.subheader('Team: Sean MacKenzie, Rami Dabit and Peter Li.')
    st.subheader('Feel free to interact with our web app.')

    # st.image('hackershrine.jpg',use_column_width=True)


def load_image(filename):
    image = cv.imread(filename)
    return image


def pinhole_camera_model():
    """
    Author: Rami Dabit
    :return:
    """
    pass


def paraxial_camera_model():
    """
    Author: Sean MacKenzie
    :return:
    """

    # optics
    f = st.slider(label='Change camera lens focal length', min_value=50, max_value=200, value=100)
    d = st.slider(label='Change camera lens diameter', min_value=25, max_value=45, value=35)

    # shape of the lens
    xcenter, ycenter = 0, 0
    width, height = 10, d
    angle = 0

    # ccd
    ccd_z = st.slider(label='Change camera sensor position', min_value=-110, max_value=-100, value=-102)
    ccd_h = st.slider(label='Change camera sensor size', min_value=2, max_value=20, value=10)

    # object
    zo = st.slider(label='Change object distance', min_value=100, max_value=10000, value=500)
    yo = st.slider(label='Change object height', min_value=-20, max_value=20, value=-15)

    def model_pinhole(zo, yo, zi):
        yi = zi * yo / zo
        return yi

    def thin_lens_model(zo, f=f):
        zi = 1 / (1 / f - 1 / zo)
        return zi

    # shape of the lens
    def add_lens_patch(width, height, xcenter=0, ycenter=0, angle=0):
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = 0.5 * width * np.cos(theta)
        y = 0.5 * height * np.sin(theta)

        rtheta = np.radians(angle)
        R = np.array([
            [np.cos(rtheta), -np.sin(rtheta)],
            [np.sin(rtheta), np.cos(rtheta)],
        ])

        x, y = np.dot(R, [x, y])
        x += xcenter
        y += ycenter

        return patches.Ellipse((xcenter, ycenter), width, height, angle=angle,
                               linewidth=0.5, fill=True, color='gray', alpha=0.125)

    def model_paraxial_lens(zo, yo, f, d, ccd_z, ccd_h):

        # in focus axial position
        zi = thin_lens_model(zo, f)

        # in focus height
        yi = model_pinhole(zo, yo, zi)

        # numerical aperture
        NA = np.arcsin(d / (2 * zo))

        # invert for plotting
        zi = -zi
        yi = -yi

        # rays - object
        ray_obj_z = [zo, 0]
        ray_obj_top = [yo, d / 2]
        ray_obj_center = [yo, 0]
        ray_obj_bottom = [yo, -d / 2]

        # rays - image
        ray_img_z = [0, zi]
        ray_img_top = [d / 2, yi]
        ray_img_center = [0, yi]
        ray_img_bottom = [-d / 2, yi]

        # rays intersecting at ccd
        ray_ccd_top = -(d / 2 - yi) * ccd_z / zi + d / 2
        ray_ccd_center = yi * ccd_z / zi
        ray_ccd_bottom = -(-d / 2 - yi) * ccd_z / zi - d / 2

        # create figure
        # fig, [ax0, ax1, ax2] = plt.subplots(ncols=3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2, 1]})
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 3, figure=fig)
        # ax0 = plt.subplot(gs[1:, :])
        ax1 = plt.subplot(gs[0, :-1])
        ax1.margins(0.01)
        ax2 = plt.subplot(gs[0, -1])
        ax1.margins(0.01)
        # ax0.margins(0.05)

        # optical axis
        ax1.axhline(0, color='black', linewidth=0.5, alpha=0.5, zorder=1.5)
        ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5, zorder=1.5)

        # lens
        for width, ax in zip([f / 10, zo / 5.555], [ax1, ax2]):
            ax.add_patch(add_lens_patch(width=width, height=d))
            ax.add_patch(add_lens_patch(width=width, height=d))

        # ccd
        ax1.plot([ccd_z, ccd_z], [-ccd_h / 2, ccd_h / 2], color='black', linewidth=3, alpha=0.25, label='CCD',
                 zorder=1.5)

        # image formation
        ax1.scatter(zi, yi, color='blue', label='image')
        ax1.plot([zi, zi], [0, yi], color='blue', linestyle='--')
        ax1.plot(ray_img_z, ray_img_top, color='blue', alpha=0.25)
        ax1.plot(ray_img_z, ray_img_center, color='blue', alpha=0.25)
        ax1.plot(ray_img_z, ray_img_bottom, color='blue', alpha=0.25)
        ax1.set_xlim([-150, 0])
        ax1.set_ylim([-25, 25])

        # object formation
        ax2.scatter(zo, yo, color='red', label='object')
        ax2.plot([zo, zo], [0, yo], color='red', linestyle='--')
        ax2.plot(ray_obj_z, ray_obj_top, color='red', alpha=0.25)
        ax2.plot(ray_obj_z, ray_obj_center, color='red', alpha=0.25)
        ax2.plot(ray_obj_z, ray_obj_bottom, color='red', alpha=0.25)
        ax2.set_xlim([0, zo * 1.25])
        ax2.set_ylim([-25, 25])
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        # rays intersecting ccd
        counter = 0
        for ray_ccd in [ray_ccd_top, ray_ccd_center, ray_ccd_bottom]:
            if ray_ccd > -ccd_h / 2 and ray_ccd < ccd_h / 2:
                if counter == 0:
                    ax1.scatter(ccd_z, ray_ccd, marker='.', color='green', alpha=0.99, label=r'$ray_{CCD}$')
                    counter = counter + 1
                else:
                    ax1.scatter(ccd_z, ray_ccd, marker='.', color='green', alpha=0.99)

        # focal plane
        ax1.plot([-f, -f], [-ccd_h / 2, ccd_h / 2], color='black', linestyle='--', alpha=0.125, label='focal plane')

        ax1.grid(alpha=0.15)
        ax1.legend()
        ax2.legend()

        plt.subplots_adjust(wspace=.001)
        plt.show()
        st.pyplot(fig=fig)

        return zi, yi, NA

    zi, yi, NA = model_paraxial_lens(zo, yo, f, d, ccd_z, ccd_h)

    image_position_string = "Image height yi = {} at axial distance zi = {}".format(np.round(-yi, 2), np.round(-zi, 2))
    numerical_aperture_string = "NA = {} degrees".format(np.round(NA * 360 / (2 * np.pi), 2))

    st.caption(body=image_position_string)
    st.caption(body=numerical_aperture_string)


def homography():
    """
    Author: Peter Li
    :return:
    """
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

    my_phi = st.slider('Change angle to decide camera position', min_value=-35, max_value=35)
    my_k = st.slider('Change Value to zoon in or zoom out', min_value=-0.2, max_value=1.0)

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

    # [Remind] use st.image to plot
    st.image(tf_img, use_column_width=True)

    # my own end
    # ========================================================


if __name__ == "__main__":
    main()