import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_subpix, corner_peaks, hessian_matrix_det
from skimage.filters import difference_of_gaussians
import pandas as pd


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Keypoints/Descriptors', 'Harris Detector', 'Hessian Detector', 'Difference of Gaussian', 'Scale-Invariant Descriptors')
    )
    
    if selected_box == 'Keypoints/Descriptors':
        keypoints_descriptors()
    if selected_box == 'Harris Detector':
        harris_detector()
    if selected_box == 'Hessian Detector':
        Hessian_detector()
    if selected_box == 'Difference of Gaussian':
        DoG()
    if selected_box == 'Scale-Invariant Descriptors':
        Scale_Invar()
 

def welcome():
    
    st.title('Feature Detection using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    st.image('Library.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image

def keypoints_descriptors():
    st.header('Keypoints and Descriptors')

    st.latex(r'''
    \text{\underline{Keypoints} are specific locations of interest in an image:}
    ''')

    st.latex(r'''
    \text{eyes, mouth, nose, mountains, buildings, or corners.}
    ''')

     
    st.latex(r'''
    \text{A \underline{keypoint descriptor} or \underline{patch} is a vector that describes the appearance of the area surrounding a keypoint.}
    ''')


    st.image('pandafeatures.png', use_column_width=True,clamp = True)

    ## gives more information about the local region surrounding the keypoint


    ##example image here

    st.latex(r'''
    \text{Keypoints and keypoint descriptors are useful for object detection, or facial recognition.}
    ''')
    st.latex(r'''
    \text{As well as for image stitching, 3D reconstruction, or camera pose estimation.}
    ''')

    ## things that require matching between various images to reach an end goal. panorama


    st.latex(r'''
    \text{...}
    ''')
    st.header('Method: Detection, Description and Matching')
    
    st.latex(r'''
    \text{1.) Find keypoints.}
    ''')
    st.latex(r'''
    \text{2.) Take patches surrounding the keypoints. i.e. keypoint descriptors}
    ''')

    st.latex(r'''
    \text{3.) Match patches between images.}
    ''')


    st.latex(r'''
    \text{...}
    ''')

    st.subheader('Basic Intuition for Keypoint Selection')

    st.latex(r'''
    \color{green}\text{Good}\color{black}\text{ keypoints should be unique and allow for easy recognition and matching across various images.}
    ''')

    

    st.latex(r'''
    \color{red}\text{Bad}\color{black}\text{ keypoints are things such as flat regions, or regions with little deviation across x and y.}
    ''')

    st.image('houseexample.png', use_column_width=True,clamp = True)
    
    st.latex(r'''
    \text{...}
    ''')
    
    st.subheader('Additional Desireable Properties')

    st.latex(r'''
    \text{We need a certain quantity of patches, to successfully match between images.}
    ''')

    st.latex(r'''
    \text{Invariant to translation, rotation, and scale.}
    ''')

    st.latex(r'''
    \text{Resistant to affine transformations.}
    ''')

    st.latex(r'''
    \text{Resistant to lighting, color, or noise variations.}
    ''')

    st.latex(r'''
    \text{...}
    ''')

    st.subheader('Now we will see some various detectors...')

def harris_detector():
    st.header("Harris Detector")

    st.latex(r'''
    \text{The basic idea behind the harris detector is that}
    ''')

    st.image('harris_example.png',use_column_width=True)

    st.latex(r'''
        \color{red}\text{a flat region:} \color{black}\text{ no change in all directions.}

        ''')

    st.latex(r'''
        \color{red}\text{ an edge:}\color{black}\text{ no change along the edge direction.}
        ''')
    
    st.latex(r'''
        \color{green}\text{ a corner:}\color{black}\text{ significant changes in all directions.}
        ''')

    st.latex(r'''...''')
    
    st.latex(r'''
    E(u,v) = \sum_{x,y}\overbrace{w(x,y)}^{\text{window function}}\, [\, \underbrace{I(x+u,y+v)}_{\text{shifted intensity}}
    - \underbrace{I(x,y)}_{\text{intensity}}\, ]^2 
     ''')
    
    st.latex(r'''...''')

    st.latex(r'''
        \text{ If we look at the second term,}
        ''')
    st.latex(r'''
    \text{for flat regions,}\, [I(x+u,y+v) -I(x,y)]^2 \approx 0
     ''')

    st.latex(r'''
    \text{ and for distinct regions,}\, [I(x+u,y+v) -I(x,y)]^2 \approx large
    ''')

    st.latex(r'''
    \text{For corner detection we wish to } \color{red}\text{maximize}\,\color{black} E(u,v)
    ''')

    st.latex(r'''\downarrow''')
    st.latex(r'''math''')
    st.latex(r'''\downarrow''')

    
    st.latex(r'''
    E(u,v) \approx  \begin{bmatrix}
                    u & v\\
                    \end{bmatrix}
                    M
                    \begin{bmatrix}
                    u\\
                    v
                    \end{bmatrix}
     ''')


    st.latex(r'''
    M=  \sum_{x,y}w(x,y)
    
                    \begin{bmatrix}
                    I_x I_x & I_x I_y\\
                    I_y I_x & I_y I_y
                    \end{bmatrix}
                    
     ''')

    st.latex(r'''
    \text{Where } Ix \text{ and } Iy \text{ are image derivatives in x and y directions.}
    ''')

    st.latex(r'''
    \text{These can be found using the sobel kernel.}
''')


    st.latex(r'''
    G_x=
    
                    \begin{bmatrix}
                    -1 & 0 & 1\\
                    -2 & 0 & 2\\
                    -1 & 0 & 1
                    \end{bmatrix},\quad
                    
                    
                

                           
    \,\,\,G_y=
                    \begin{bmatrix}
                     1 & 2 & 1\\
                     0 & 0 & 0\\
                    -1 & -2 & -1
                    \end{bmatrix}

                                  
     ''')

    st.latex(r'''...''')

    st.latex(r'''
    \text{A scoring function R is created, which determines if a corner is captured in a window}
    ''')

    st.latex(r'''
   R = det\,M-k(\,\,Tr[M]\,\,)^2     
     ''')

    st.latex(r''' \quad det\,M = \lambda_1 \lambda_2 \quad \textrm{\&} \quad  Tr[M] = \lambda_1 + \lambda_2       
     ''')

    st.latex(r'''...''')

    st.latex(r'''
\text{Thresholding to R:}''')

    st.image('eigenvalues.png', use_column_width=True,clamp = True)

    st.latex(r'''
   \text{R}\approx \text{small} \implies \color{red}\text{flat region}      
     ''')
    st.latex(r'''
   \text{R}< 0 \implies \color{red}\text{edge}      
     ''')
    st.latex(r'''
   \text{R}\approx{large}\implies \color{green}\text{corner}      
     ''')


    
    

    filename = st.selectbox(
     'Which image do you want to process?',
     ('UCSB_Henley_gate.jpg', 'Building.jpeg', 'checkerboard.png','Library.jpg'))

    # sliders ------------------------------------------------------------------------------------------

    thresh = st.slider('Change Threshold', min_value=0.0000, max_value=.5000,step=0.0001, format='%f')

    block_size = st.slider('Change Block Size', min_value=2, max_value=10)

    aperture_size = st.slider('Change Aperture', min_value=1, max_value=31,step=2)

    k = st.slider('Harris Detector Free Variable', min_value=0.0000, max_value=.1000,step=0.0001,value=0.04, format='%f')

    iteration_count = st.slider('Change Dilation', min_value=1, max_value=100, value=2)

    # harris detector processing ------------------------------------------------------------------------
    img = cv2.imread(filename)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, block_size, aperture_size, k)

    # dilation of the points
    dst = cv2.dilate(dst, None, iterations=iteration_count)

    # Thresholding
    img[dst > thresh * dst.max()] = [0,0,255]
    st.image(img, use_column_width=True,channels="BGR")

def Hessian_detector():
    #Andrew Yung
    st.header("Feature Detection with Hessian Detector")
    st.subheader("How it works:")
    st.write("1. The Hessian of the image corresponds to the curvature of the image based on its pixel values.")

    st.latex(r'''
    H(I) = \begin{bmatrix}
    I_{xx} & I_{xy} \\
    I_{xy} & I_{yy}
    \end{bmatrix}
     ''')
        
    st.write("2. When we perform the eigenvalue decomposition of H(I)(x,y), the eigenvectors correspond to the direction of greatest and lowest curvature and their respective eigenvalues correspond to the magnitude of curvature")
    st.latex(r'''
    eig(H(I)) = \left\{
        \begin{array}{ll}
            \underline{e_1} , \lambda_1 \text{=> Greatest curvature}\\
            \underline{e_2} , \lambda_2 \text{=>Lowest curvature}
        \end{array}
    \right.
    ''')

    st.write("3. Since we are only interested in the strength of curvature we can simply take the determinant of H to yield the overall curvature strength for all x,y coordinates")
    st.latex(r'''
    det(H) => \lambda_1 * \lambda_2
    ''')
    st.write("4. Threshold the determinant \"image\" to yield our coordinate features!")

    st.subheader("Hessian Detector Demo")
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if image_file is not None:
        image = Image.open(image_file)
        img = np.array(image)
        img_rgb = img
    else:
        img = load_image('Banff.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y = img_gray.shape
    rad = int(0.0065 * x)
    max_dis= 10*int( 0.004 *x)

    thres = st.slider('Change Threshold value',min_value = 0.01,max_value = 0.5, value=0.05)
    min_dis = st.slider('Change Minimum Distance',min_value = 1,max_value = max_dis)
    

    #st.image(img_gray, use_column_width=True)
    dets = hessian_matrix_det(img_gray)
    #st.image(-dets, clamp = True, channels = 'gray')

    coords_hessian = corner_peaks(hessian_matrix_det(img_gray), min_distance=min_dis, threshold_rel=thres)

    st.text("Hessian Features Detected")
    
    HesImg = img_rgb
    for (y,x) in coords_hessian:
        HesImg = cv2.circle(HesImg, (x,y), radius=rad, color=(255,0,0), thickness=-1)
    st.image(HesImg, use_column_width=True,clamp = True)
    

def sigmoid(x,s):
    #Andrew Yung
    if (s == 0):
        l = len(x)
        s = np.zeros(l)
        hf= l//2
        s[hf:l] = 1
        sig = s
    else:
        z = np.exp(-x/s)
        sig = 1 / (1 + z)
    
    return sig

def DoG():
    ## Andrew Yung
    st.header("Difference of Gaussian Detector")
    st.subheader("How it works:")
    st.write("1. We take two blurred versions of the image w.r.t two sigmas")
    sig0 = st.slider('Select a sigmas', 0.0, 10.0, (0.0, 0.0))
    st.write("2. We subtract the two blurred images and yield a bandpass filterd image")
    x = np.arange(-5,5,0.01, dtype = float)
    s0 = sigmoid(x,0)
    s1 = sigmoid(x,sig0[0])
    s2 = sigmoid(x,sig0[1])
    s3 = s2-s1
    s = np.stack((s0,s1,s2,s3),axis=1)
    
    df = pd.DataFrame(s, columns=['Edge','s1','s2',"s2-s1"])
    st.line_chart(df)
    st.write("3. We threshold the new image to yield our feature points/edges")

    

    st.subheader('Difference of Gaussian in images')
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if image_file is not None:
        image = Image.open(image_file)
        img = np.array(image)
        img_rgb = img
    else:
        img = load_image('jerry.jpg')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    x,_ = img_gray.shape
    rad = int(0.007 * x)
    max_dis= 10*int( 0.004 *x)

    thres = st.slider('Change Threshold value',min_value = 0.01,max_value = 1.0)
    min_dis = st.slider('Change Minimum Distance',min_value = 1,max_value = max_dis)
    sig = st.slider('Select a sigmas', 0.0, 50.0, (2.0, 10.0))
    dog = difference_of_gaussians(image=img_gray, low_sigma=sig[0], high_sigma=sig[1], channel_axis=-1)
    norm_image = cv2.normalize(dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    st.image(norm_image,use_column_width=True,clamp = True)
    coords_dog = corner_peaks(dog, min_distance=min_dis, threshold_rel=thres)

    DogImg = img_rgb
    for (y,x) in coords_dog:
        DogImg = cv2.circle(DogImg, (x,y), radius=rad, color=(255,0,0), thickness=-1)
    st.image(DogImg, use_column_width=True,clamp = True)

def Scale_Invar():
    ##Mason Corey
    st.header('Scale-Invariant Detectors')
    st.subheader("Why do we need Scale Invariance?")
    st.write('Common detectors, like the Harris and Hessian detectors, are often invariant to things like illumination, translation, and rotation, but not scaling.')
    intro1 = load_image("scale_inv_intro1.png")
    intro1_rgb = cv2.cvtColor(intro1, cv2.COLOR_BGR2RGB)
    st.image(intro1_rgb)
    st.text('')

    st.write('Regions of different sizes will look the same in two images that only differ in scaling. How can we make a detector that will find the same keypoints independently in two images with different scale?')
    intro2 = load_image("scale_inv_intro2.png")
    intro2_rgb = cv2.cvtColor(intro2, cv2.COLOR_BGR2RGB)
    st.image(intro2_rgb)
    st.text('')

    st.subheader('Naive Approach')
    st.write('The Naive Approach is to take two images that differ only in scale, compute the Gaussian pyramid for both, and do NxN pairwise comparisons to match similar pyramids and determine the relative scale for keypoint detection.')
    st.markdown("""
    * Drawbacks:
        * Very computationally expensive
        * Requires more than one image to compare
    """)
    intro3 = load_image("scale_inv_intro3.png")
    intro3_rgb = cv2.cvtColor(intro3, cv2.COLOR_BGR2RGB)
    st.image(intro3_rgb)
    st.text('')

    st.subheader('More Robust Solution: The Laplacian Pyramid')
    st.write('We want to generate keypoints that will be found in the same location regardless of scale and can be found independently of other images (i.e. no comparison required).')
    st.write('To do this, we need to find a function to apply to the image that has some point which is identifiable regardless of scale, which we can set as a keypoint.')
    st.write('The easiest function to use is one with a single maximum peak. The maximum will not change with scale, so we can use the maximum point as our keypoint.')
    intro4 = load_image("scale_inv_intro4.png")
    intro4_rgb = cv2.cvtColor(intro4, cv2.COLOR_BGR2RGB)
    st.image(intro4_rgb)
    intro5 = load_image("scale_inv_intro5.png")
    intro5_rgb = cv2.cvtColor(intro5, cv2.COLOR_BGR2RGB)
    st.image(intro5_rgb)
    st.text('')
    st.write('The most ideal function that matches these characteristics is the Laplacian Pyramid, which can be quickly approximated using the Difference of Gaussians:')
    intro6 = load_image("scale_inv_intro6.png")
    intro6_rgb = cv2.cvtColor(intro6, cv2.COLOR_BGR2RGB)
    st.image(intro6_rgb, width=500)
    st.text('')
    st.write('You can then find the characteristic scale for each keypoint, which is the scale that produces the peak response for the Derivative of Gaussian of the image in the area of the keypoint.')
    st.write('The characteristic scale for a given keypoint will give the best invariance to scale for that keypoint.')
    intro7 = load_image("scale_inv_intro7.png")
    intro7_rgb = cv2.cvtColor(intro7, cv2.COLOR_BGR2RGB)
    st.image(intro7_rgb, width=500)
    intro8 = load_image("scale_inv_intro8.JPG")
    intro8_rgb = cv2.cvtColor(intro8, cv2.COLOR_BGR2RGB)
    st.image(intro8_rgb)
    st.text('')

    st.subheader('Implementation of Scale-Invariant Detection in Industry')
    st.markdown("""
    * There are two common implementations of Scale-Invariant Detectors:
        * Harris-Laplacian Detection
        * SIFT (Scale-Invariant Feature Transformation)
    """)
    intro9 = load_image("scale_inv_intro9.png")
    intro9_rgb = cv2.cvtColor(intro9, cv2.COLOR_BGR2RGB)
    st.image(intro9_rgb)
    st.write('Here is an example of the Laplacian Pyramid being generated with Gaussian blur being applied with a progressively-increasing sigma:')
    intro10 = load_image("scale_inv_intro10.png")
    intro10_rgb = cv2.cvtColor(intro10, cv2.COLOR_BGR2RGB)
    st.image(intro10_rgb)
    st.write('Here is the Difference of Gaussian for the first octave:')
    intro11 = load_image("scale_inv_intro11.png")
    intro11_rgb = cv2.cvtColor(intro11, cv2.COLOR_BGR2RGB)
    st.image(intro11_rgb)

    st.subheader('SIFT Demo')
    #Demo here
    img = cv2.imread('sift_img.jpg')
    dim = (img.shape[1],img.shape[0])

    #Make a smaller copy
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim_small = (width, height)
    img_small = cv2.resize(img, dim_small, interpolation = cv2.INTER_NEAREST)

    #Make a larger copy
    scale_percent = 140 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim_large = (width, height)
    img_large = cv2.resize(img, dim_large, interpolation = cv2.INTER_NEAREST)

    #Make grayscale versions
    gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    
    num_features = st.slider('Number of Features to Retain', min_value = 10, max_value = 10000, value = 1000)
    num_octaves = st.slider('Number of Octaves', min_value = 1, max_value = 20, value = 6)
    contrast_thresh = st.slider('Contrast Threshold for Filtering Weak Features in Low-Contrast Regions', min_value = 0.01, max_value = 0.1, value = 0.04)
    edge_thresh = st.slider('Threshold for Filtering Weak Edges', min_value = 1, max_value = 100, value = 10)
    sigma = st.slider('Initial Sigma', min_value = 0.5, max_value = 5.0, value = 1.6)
    sift = cv2.SIFT_create(num_features, num_octaves, contrast_thresh, edge_thresh, sigma)

    kp_small = sift.detect(gray_small,None)
    for kp in kp_small:
        kp.size *= .6
    img_small = cv2.drawKeypoints(gray_small,kp_small,img_small, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_small = cv2.resize(img_small,dim,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('SIFT_keypoints_small.jpg',img_small)

    kp = sift.detect(gray,None)
    img = cv2.drawKeypoints(gray,kp,img, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('SIFT_keypoints.jpg',img)

    kp_large = sift.detect(gray_large,None)
    for kp in kp_large:
        kp.size *= 1.4
    img_large = cv2.drawKeypoints(gray_large,kp_large,img_large, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_large = cv2.resize(img_large,dim,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('SIFT_keypoints_large.jpg',img_large)

    st.subheader("0.6x Scaled Image")
    st.image('SIFT_keypoints_small.jpg')
    st.text('')

    st.subheader("Original Image")
    st.image('SIFT_keypoints.jpg')
    st.text('')

    st.subheader("1.4x Scaled Image")
    st.image('SIFT_keypoints_large.jpg')
    st.text('')

if __name__ == "__main__":
    main()
