# notes

# imports
import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np

from skimage import data, util, io
from skimage.exposure import rescale_intensity
from skimage import data
from skimage import transform
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

from os import path
from scipy import optimize as opt
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
        camera_intrinsics()


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
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

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


def camera_intrinsics():
    """
    Author: Sean MacKenzie
    References:
    [1] Zhang's camera calibration
    [2] Burger's
    [3] Blog
    :return:
    """

    # chessboard pattern and size
    pattern_dim = (4, 5)
    square_dim = 1.0

    def generate_synthetic_chessboards(image_number=1, save_image=False, scale=1.0, rotate_degrees=0, shear=0.0,
                                       translate_x=0, translate_y=0):

        # sizes
        cb_size_x = 149  # checkerboard: number of rows, where one row is 30 pixels wide (range: 75:25:175)
        cb_size_y = 124  # checkerboard: number of columns, where one columns is 30 pixels tall (range: 75:25:175)
        cb_shape_x = 512  # image shape: number of columns
        cb_shape_y = 512  # image shape: number of rows

        # transformations
        #scale = 1.8
        scale_x = scale  # stretch in +x-dir. (do not use - unrealistic stretching)
        scale_y = scale  # stretch in +y-dir. (do not use - unrealistic stretching)

        #rotate_degrees = -30  # rotation in clockwise-dir. (range: 0:360)
        rotation = rotate_degrees * 2 * np.pi / 360

        #shear = 0.2  # shear in clockwise-dir. (range: -0.8:0.1:0.8)

        #translate_x = -100  # translation in +x-dir. (range: 0:cb_shape_x - cb_size_x)
        #translate_y = 40  # translation in +y-dir. (range: 0:cb_shape_y - cb_size_y)
        trans_x = cb_shape_y // 2 - cb_size_y // 2 + translate_x
        trans_y = cb_shape_x // 2 - cb_size_x // 2 + translate_y

        # Transformed checkerboard
        tform = AffineTransform(scale=(scale_x, scale_y), rotation=rotation, shear=shear,
                                translation=(trans_x, trans_y))
        image = warp(data.checkerboard()[:cb_size_x, :cb_size_y], tform.inverse, output_shape=(cb_shape_y, cb_shape_x))

        # rescale to 16-bit
        image = rescale_intensity(image, out_range=np.uint16)

        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        ax.axis((0, cb_shape_x, cb_shape_y, 0))
        plt.show()
        st.image(image, use_column_width=True)

        if save_image:
            io.imsave('syn_chessboard_4x4_{}.tif'.format(image_number), image)

    def get_camera_images():
        images = ['syn_chessboard_4x4_{}.tif'.format(each) for each in np.arange(1, 5)]
        images = sorted(images)
        for each in images:
            yield (each, cv.imread(each, 0))

    def getChessboardCorners(images=None, visualize=False):
        objp = np.zeros((pattern_dim[1] * pattern_dim[0], 3), dtype=np.float64)
        objp[:, :2] = np.indices(pattern_dim).T.reshape(-1, 2)
        objp *= square_dim

        chessboard_corners = []
        image_points = []
        object_points = []
        correspondences = []
        ctr = 0
        for (path, each) in get_camera_images():  # images:
            print("Processing Image : ", path)
            print(each.shape)
            print(np.max(each))

            if np.mean(each) < np.max(each // 2):
                each = cv.bitwise_not(each)

            ret, corners = cv.findChessboardCorners(each, patternSize=pattern_dim)
            if ret:
                print("Chessboard Detected ")
                corners = corners.reshape(-1, 2)

                if corners.shape[0] == objp.shape[0]:
                    image_points.append(corners)
                    object_points.append(objp[:,
                                         :-1])  # append only World_X, World_Y. Because World_Z is ZERO. Just a simple modification for get_normalization_matrix
                    assert corners.shape == objp[:, :-1].shape, "mismatch shape corners and objp[:,:-1]"
                    correspondences.append([corners.astype(int), objp[:, :-1].astype(int)])

                if visualize:
                    # Draw and display the corners
                    ec = cv.cvtColor(each, cv.COLOR_GRAY2BGR)
                    cv.drawChessboardCorners(ec, pattern_dim, corners, ret)

                    # to show via skimage
                    fig, ax = plt.subplots()
                    ax.imshow(ec)
                    plt.show()
                    st.image(ec, use_column_width=True)
            else:
                print("Error in detection points", ctr)

            ctr += 1

        return correspondences

    def compute_view_based_homography(correspondence, reproj=False):
        """
        correspondence = (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv)
        """
        image_points = correspondence[0]
        object_points = correspondence[1]
        normalized_image_points = correspondence[2]
        normalized_object_points = correspondence[3]
        N_u = correspondence[4]
        N_x = correspondence[5]
        N_u_inv = correspondence[6]
        N_x_inv = correspondence[7]

        N = len(image_points)
        print("Number of points in current view : ", N)

        M = np.zeros((2 * N, 9), dtype=np.float64)
        print("Shape of Matrix M : ", M.shape)

        print("N_model\n", N_x)
        print("N_observed\n", N_u)

        # create row wise allotment for each 0-2i rows
        # that means 2 rows..
        for i in range(N):
            X, Y = normalized_object_points[i]  # A
            u, v = normalized_image_points[i]  # B

            row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
            row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])
            M[2 * i] = row_1
            M[(2 * i) + 1] = row_2

            print("p_model {0} \t p_obs {1}".format((X, Y), (u, v)))

        # M.h  = 0 . solve system of linear equations using SVD
        u, s, vh = np.linalg.svd(M)
        #print("Computing SVD of M")
        #print("U : Shape {0} : {1}".format(u.shape, u))
        #print("S : Shape {0} : {1}".format(s.shape, s))
        #print("V_t : Shape {0} : {1}".format(vh.shape, vh))
        #print(s, np.argmin(s))

        h_norm = vh[np.argmin(s)]
        h_norm = h_norm.reshape(3, 3)

        print("Normalized Homography Matrix : \n", h_norm)

        #print(N_u_inv)
        #print(N_x)

        # h = h_norm
        h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

        # if abs(h[2, 2]) > 10e-8:
        h = h[:, :] / h[2, 2]

        # print("Homography for View : \n", h)

        if reproj:
            reproj_error = 0
            for i in range(len(image_points)):
                t1 = np.array([[object_points[i][0]], [object_points[i][1]], [1.0]])
                t = np.matmul(h, t1).reshape(1, 3)
                t = t / t[0][-1]
                formatstring = "Imp {0} | ObjP {1} | Tx {2}".format(image_points[i], object_points[i], t)
                print(formatstring)
                reproj_error += np.sum(np.abs(image_points[i] - t[0][:-1]))
            reproj_error = np.sqrt(reproj_error / N) / 100.0
            print("Reprojection error : ", reproj_error)

        return h

    def normalize_points(chessboard_correspondences):
        views = len(chessboard_correspondences)

        def get_normalization_matrix(pts, name="A"):
            pts = pts.astype(np.float64)
            x_mean, y_mean = np.mean(pts, axis=0)
            var_x, var_y = np.var(pts, axis=0)

            s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)

            # print("Matrix: {4} : meanx {0}, meany {1}, varx {2}, vary {3}, sx {5}, sy {6} ".format(x_mean, y_mean, var_x, var_y, name, s_x, s_y))

            n = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])
            # print(n)

            n_inv = np.array([[1. / s_x, 0, x_mean], [0, 1. / s_y, y_mean], [0, 0, 1]])
            return n.astype(np.float64), n_inv.astype(np.float64)

        ret_correspondences = []
        for i in range(views):
            imp, objp = chessboard_correspondences[i]
            N_x, N_x_inv = get_normalization_matrix(objp, "A")
            N_u, N_u_inv = get_normalization_matrix(imp, "B")

            # convert imp, objp to homogeneous
            hom_imp = np.array([[[each[0]], [each[1]], [1.0]] for each in imp])
            hom_objp = np.array([[[each[0]], [each[1]], [1.0]] for each in objp])

            normalized_hom_imp = hom_imp
            normalized_hom_objp = hom_objp

            for i in range(normalized_hom_objp.shape[0]):
                # 54 points iterate one by one & all points are homogeneous
                n_o = np.matmul(N_x, normalized_hom_objp[i])
                normalized_hom_objp[i] = n_o / n_o[-1]

                n_u = np.matmul(N_u, normalized_hom_imp[i])
                normalized_hom_imp[i] = n_u / n_u[-1]

            normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
            normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

            normalized_objp = normalized_objp[:, :-1]
            normalized_imp = normalized_imp[:, :-1]

            ret_correspondences.append((imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))

        return ret_correspondences

    def minimizer_func(initial_guess, X, Y, h, N):
        # X : normalized object points flattened
        # Y : normalized image points flattened
        # h : homography flattened
        # N : number of points

        x_j = X.reshape(N, 2)
        # Y = Y.reshape(N, 2)
        # h = h.reshape(3, 3)

        projected = [0 for i in range(2 * N)]
        for j in range(N):
            x, y = x_j[j]
            w = h[6] * x + h[7] * y + h[8]

            projected[2 * j] = (h[0] * x + h[1] * y + h[2]) / w
            projected[2 * j + 1] = (h[3] * x + h[4] * y + h[5]) / w

        return (np.abs(projected - Y)) ** 2

    def jac_function(initial_guess, X, Y, h, N):
        x_j = X.reshape(N, 2)
        jacobian = np.zeros((2 * N, 9), np.float64)
        for j in range(N):
            x, y = x_j[j]
            sx = np.float64(h[0] * x + h[1] * y + h[2])
            sy = np.float64(h[3] * x + h[4] * y + h[5])
            w = np.float64(h[6] * x + h[7] * y + h[8])
            jacobian[2 * j] = np.array([x / w, y / w, 1 / w, 0, 0, 0, -sx * x / w ** 2, -sx * y / w ** 2, -sx / w ** 2])
            jacobian[2 * j + 1] = np.array(
                [0, 0, 0, x / w, y / w, 1 / w, -sy * x / w ** 2, -sy * y / w ** 2, -sy / w ** 2])

        return jacobian

    def refine_homographies(H, correspondence, skip=False):
        if skip:
            return H

        image_points = correspondence[0]
        object_points = correspondence[1]
        normalized_image_points = correspondence[2]
        normalized_object_points = correspondence[3]
        N_u = correspondence[4]
        N_x = correspondence[5]
        N_u_inv = correspondence[6]
        N_x_inv = correspondence[7]

        N = normalized_object_points.shape[0]
        X = object_points.flatten()
        Y = image_points.flatten()
        h = H.flatten()
        h_prime = opt.least_squares(fun=minimizer_func, x0=h, jac=jac_function, method="lm", args=[X, Y, h, N],
                                    verbose=0)

        if h_prime.success:
            H = h_prime.x.reshape(3, 3)
        H = H / H[2, 2]
        return H

    def get_intrinsic_parameters(H_r):
        M = len(H_r)
        V = np.zeros((2 * M, 6), np.float64)

        def v_pq(p, q, H):
            v = np.array([
                H[0, p] * H[0, q],
                H[0, p] * H[1, q] + H[1, p] * H[0, q],
                H[1, p] * H[1, q],
                H[2, p] * H[0, q] + H[0, p] * H[2, q],
                H[2, p] * H[1, q] + H[1, p] * H[2, q],
                H[2, p] * H[2, q]
            ])
            return v

        for i in range(M):
            H = H_r[i]
            V[2 * i] = v_pq(p=0, q=1, H=H)
            V[2 * i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

        # solve V.b = 0
        u, s, vh = np.linalg.svd(V)
        b = vh[np.argmin(s)]

        # print("V.b = 0 Solution : ", b.shape)

        # according to zhangs method
        vc = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)
        l = b[5] - (b[3] ** 2 + vc * (b[1] * b[2] - b[0] * b[4])) / b[0]
        alpha = np.sqrt((l / b[0]))
        beta = np.sqrt(((l * b[0]) / (b[0] * b[2] - b[1] ** 2)))
        gamma = -1 * ((b[1]) * (alpha ** 2) * (beta / l))
        uc = (gamma * vc / beta) - (b[3] * (alpha ** 2) / l)

        print([vc,
               l,
               alpha,
               beta,
               gamma,
               uc])

        A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
        print("Intrinsic Camera Matrix is :")
        print(A)
        return A

    chessboard_correspondences = getChessboardCorners(images=None, visualize=True)

    chessboard_correspondences_normalized = normalize_points(chessboard_correspondences)

    print("M = ", len(chessboard_correspondences_normalized), " view images")
    print("N = ", len(chessboard_correspondences_normalized[0][0]), " points per image")

    H = []
    for correspondence in chessboard_correspondences_normalized:
        H.append(compute_view_based_homography(correspondence, reproj=0))

    H_r = []
    for i in range(len(H)):
        h_opt = refine_homographies(H[i], chessboard_correspondences_normalized[i], skip=False)
        H_r.append(h_opt)

    A = get_intrinsic_parameters(H_r)






if __name__ == "__main__":
    main()