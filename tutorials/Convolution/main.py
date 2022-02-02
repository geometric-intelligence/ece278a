# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 2022
​
@authors: Abhijith, Canan, Sansar
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import math


def main():
    
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','What is Convolution?','Blurring Kernel','Sharpening Kernel', 
     'Edge Detector', 'Gaussian Kernel','Sobel Kernel','Corner Detector')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'What is Convolution?':
        what_is_convolution()    
    if selected_box == 'Gaussian Kernel':
        gausian_kernel()
    if selected_box == 'Blurring Kernel':
        blurring_kernel()
    if selected_box == 'Sobel Kernel':
        sobel_kernel()
    if selected_box == 'Edge Detector':
        edge_detector_kernel()
    if selected_box == 'Corner Detector':
        corner_detector_kernel()
    if selected_box == 'Sharpening Kernel':
        sharpen_kernel()

def welcome():
    
    st.title('Convolution in Image Processing')
    
    st.subheader('By Abhijith Atreya, Canan Cebeci, and Sansar Yogi')

    img = Image.open('conv.png')
    display_image(img)
    st.write('Source: https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37')



def what_is_convolution():
    st.title('What is Convolution?')

    st.write('Convolution is a mathematical operation between two signals. For 2-D signals (i.e. images), it is defined as follows:')

    st.latex(r'''
    g[i,j] = (f*h)[i,j] = \sum_{u=-k}^{k} \sum_{v=-l}^{l} f[u,v]h[i-u,j-v] 
     ''')

    st.write('This operation is a neighborhood operator that can be thought of as the cross-correlation between $f[i,j]$ and $h[-i,-j]$. In other words, the output of a convolution at a certain location is an inner product between the first signal and a flipped-and-shifted version of the second.')

    st.write('Consider the following example of a convolution between two matrices A and B, given by')

    st.latex(r'''
    A = \begin{bmatrix}
1 & -1 & 2\\
3 & 1 & -4\\
-5 & 0 & -2
\end{bmatrix},
    B = \begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
     ''')

    st.write('To compute this convolution, we will first flip the B matrix both horizontally and vertically, resulting in')

    st.latex(r'''
        B_{flip} = \begin{bmatrix}
4 & 3\\
2 & 1
\end{bmatrix}
    ''')

    st.write('Now, we compute output pixels by sliding $B_{flip}$ across $A$ and computing the inner product. We find that:')

    st.latex(r'''
    A * B = \begin{bmatrix}
C_{11} & C_{12}\\
C_{21} & C_{22}
\end{bmatrix}
    ''')
    st.latex(r'''
    C_{11} = \begin{bmatrix}
1 & -1\\
3 & 1
\end{bmatrix} \cdot \begin{bmatrix}
4 & 3\\
2 & 1
\end{bmatrix} = (1)(4) + (-1)(3) + (3)(2) + (1)(1) = 8
    ''')
    st.latex(r'''
    C_{12} = \begin{bmatrix}
-1 & 2\\
1 & -4
\end{bmatrix} \cdot \begin{bmatrix}
4 & 3\\
2 & 1
\end{bmatrix} = (-1)(4) + (2)(3) + (1)(2) + (-4)(1) = 0
    ''')
    st.latex(r'''
    C_{21} = \begin{bmatrix}
3 & 1\\
-5 & 0
\end{bmatrix} \cdot \begin{bmatrix}
4 & 3\\
2 & 1
\end{bmatrix} = (3)(4) + (1)(3) + (-5)(2) + (0)(1) = 5
    ''')
    st.latex(r'''
    C_{22} = \begin{bmatrix}
1 & -4\\
0 & -2
\end{bmatrix} \cdot \begin{bmatrix}
4 & 3\\
2 & 1
\end{bmatrix} = (1)(4) + (-4)(3) + (0)(2) + (-2)(1) = -10
    ''')

    st.write('In this example, note that the output image was smaller than the original. This will always happen if the kernel ($B_{flip}$) does not slide outside the input image to cover the edges.')

    st.write('For image processing applications, we typically want the output image to be the same size. To accomplish this, there must be a border of extra pixels added to the input image. There are multiple options for that, but the convolutions shown today will be padded using an interpolation method.')

    st.header('LSI Property')

    st.write('The convolution operator can be shown to be a linear and shift-invariant operator. Because of this, the operator inherits some key properties of LSI operators')

    st.write('Two key properties that are often used to derive many kernels, including a few shown in this app, are the Commutative and Associative Properties which allow us to write')

    st.latex(r'''
    F*H = H*F \\
    (F*H_1)*H_2 = F*(H_1*H_2) \\
    ''')

    st.write('In other words, if we wish to apply multiple kernels in succession we can do so by convolving the kernels with each other, in any order, and then convolve the image with the resulting kernel.')

    st.write('We also have the Distributive Property, which states that') 

    st.latex(r'''
    F*(H_1 + H_2) = F*H_1 + F*H_2 
    ''')

def load_image(filename):
    image = cv2.imread(filename)
    return image

def display_image(op):
    st.image(op, use_column_width=True,clamp = True)
    
def do_convolution(img, op, kernel):
    
    #rgb channels
    kernel = cv2.flip(kernel,-1)
    op1 = cv2.filter2D(img[:,:,0],-1,kernel)
    op2 = cv2.filter2D(img[:,:,1],-1,kernel)
    op3 = cv2.filter2D(img[:,:,2],-1,kernel)
    # combine the channels
    op[...,0] = op1
    op[...,1] = op2
    op[...,2] = op3
    return op

def do_convolution_norm(img, op, kernel):
    
    #rgb channels
    op1 = cv2.filter2D(img[:,:,0],-1,kernel)
    op2 = cv2.filter2D(img[:,:,1],-1,kernel)
    op3 = cv2.filter2D(img[:,:,2],-1,kernel)
    # combine the channels
    op[...,0] = np.multiply(op1, 255.0/np.amax(op1))
    op[...,1] = np.multiply(op2, 255.0/np.amax(op2))
    op[...,2] = np.multiply(op3, 255.0/np.amax(op3))
    return op

def gausian_kernel():
    #Gaussian kernel
    st.header('Gaussian Kernel')

    st.subheader('Kernel Design')

    st.write('The Gaussian Kernel is shown below in its 5x5 example:')

    st.latex(r'''
    \frac{1}{256} \begin{bmatrix}
1 & 4 & 6 & 4 & 1\\
4 & 16 & 24 & 16 & 4\\
6 & 24 & 36 & 24 & 6\\
4 & 16 & 24 & 16 & 4\\
1 & 4 & 6 & 4 & 1 
\end{bmatrix} = \frac{1}{16} \begin{bmatrix}
 1 & 4 & 6 & 4 & 1 \end{bmatrix} * \frac{1}{16}
 \begin{bmatrix}
 1\\4\\6\\4\\1 \end{bmatrix}
    ''')

    st.write('Notice that it is derived by convolving a linear tent vector with a transposed version of itself. This allows it to function as a smoothing kernel in all directions.')

    st.write('We can convolve an image with a Gaussian kernel to see its effect...')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)

        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_len1 = np.shape(img)[0]
        img_len2 = np.shape(img)[1]
        st.subheader("Convolution with Gaussian kernel")
        n = st.slider('Change kernel size of Gaussian kernel',min_value = 7,max_value = 31,step=4)
        sigma = st.slider('Change sigma for Gaussian',min_value = 1,max_value = 10,step=1)
        # Gaussian kernel generation
        kernel_gaus = np.zeros((n,n),np.float32)
        origin = (n-1)/2
        for i in range(n):
            for j in range(n):
                kernel_gaus[i][j] = (1/(2*pow(math.pi*sigma,2)))*math.exp( -(pow(i-origin,2) +pow(j-origin,2))/ (2*pow(sigma,2)))
        op_gauss = np.zeros((img_len1,img_len2,3), 'uint8')
        op_gauss = do_convolution_norm(img,op_gauss,kernel_gaus)
        display_image(op_gauss)
        if st.button('See the Gaussian Kernel'):
            st.text(kernel_gaus)
        st.markdown("***")

def blurring_kernel():
    st.header('Blurring Kernel')

    st.subheader('Kernel Design')

    st.write('The Blurring, or Moving Average Kernel consists of all 1s, divided by the size of kernel. 3x3 example shown below:')

    st.latex(r'''
    \frac{1}{9} \begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix} 
    ''')

    st.write('Note that this kernel essentially takes the average of pixel values within a neighborhood, making it a smoothing kernel.')

    st.write('We can convolve an image with a Blurring kernel to see its effect...')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)

        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_len1 = np.shape(img)[0]
        img_len2 = np.shape(img)[1]
        op_blur = np.zeros((img_len1,img_len2,3), 'uint8')
        #box blurring
        st.subheader("Convolution with blurring filter")
        x = st.slider('Change Threshold value for blurring',min_value = 5,max_value = 25) 
        #blur kernel
        kernel_blur = np.ones((x,x),np.float32)/(x*x)
        op_blur = do_convolution(img,op_blur,kernel_blur)
        display_image(op_blur)
        if st.button('See the Blurring Kernel'):
            st.text(kernel_blur)
        st.markdown("***")

def sobel_kernel():
    #Sobel kernel
    st.header('Sobel Kernel')

    st.subheader('Kernel Design')

    st.write('The Sobel Operator is shown below, both the horizontal and vertical edge versions:')

    st.latex(r'''
    S_x = \begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{bmatrix}, 
    S_y = \begin{bmatrix}
1 & 2 & 1\\
0 & 0 & 0\\
-1 & -2 & -1
\end{bmatrix}
    ''')

    st.write('Both of these kernels are outputting pixel values based on the differnece between surrounding pixels (to the left/right for $S_x$ and above/below for $S_y$), making them edge detectors.')

    st.write('We can convolve an image with a Sobel kernel to see its effect...')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)

        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.subheader("Convolution with Sobel kernels") 
        kernel_sobelx = np.array([[-1,0,1], [-2, 0,2], [-1,0,1]])
        kernel_sobely = np.array([[1,2,1], [0, 0,0], [-1,-2,-1]])
        opx= cv2.filter2D(img_gray,-1,kernel_sobelx)
        opy= cv2.filter2D(img_gray,-1,kernel_sobely)
        op_sobel = np.sqrt(pow(opx,2) +pow(opy,2))
        st.text('Vertical edges:')
        st.image(opx, use_column_width=True,clamp = True)
        if st.button('See the Sobel Kernel for horizontal intensity change (vertical edges)'):
            st.text(kernel_sobelx)
        st.text('Horizontal edges:')
        st.image(opy, use_column_width=True,clamp = True)
        if st.button('See the Sobel Kernel for vertical intensity change (horizontal edges)'):
            st.text(kernel_sobely)
        st.text('Vertical and horizontal edges combined:')
        st.image(op_sobel, use_column_width=True,clamp = True)
        st.markdown("***")

def edge_detector_kernel():
    st.header('Edge Detector kernel')

    st.subheader('Kernel Design')
    st.write('The edge detector kernel represents object boundaries. It is also called a laplacian edge detector. It performs a second order derivative on the image. The kernel being used is shown below:')

    st.latex(r'''
    \begin{bmatrix}
-2 & -2 & -2\\
-2 &  16 & -2\\
-2 & -2 & -2
\end{bmatrix}, 
    ''')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)
        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_len1 = np.shape(img)[0]
        img_len2 = np.shape(img)[1]
        #edge detection
        st.subheader("Convolution with edge detection kernel")
        kernel_edge = np.array([[-2,-2,-2], [-2, 16,-2], [-2,-2,-2]])
        op_edge = np.zeros((img_len1,img_len2,3), 'uint8')
        op_edge = do_convolution(img,op_edge, kernel_edge)
        display_image(op_edge)
        if st.button('See the Laplacian Kernel'):
            st.text(kernel_edge)
        st.markdown("***")

def corner_detector_kernel():
    st.header('Corner detector kernel')

    st.subheader('Kernel Design')
    st.write('The corner detector detects image corners. The kernel is shown below:')

    st.latex(r'''
    \begin{bmatrix}
1 & -2 & 1\\
-2 &  4 & -2\\
1 & -2 & 1
\end{bmatrix}, 
    ''')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)

        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_len1 = np.shape(img)[0]
        img_len2 = np.shape(img)[1]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        op_gray = np.zeros((img_len1,img_len2), 'uint8')
        #corner detection
        st.subheader("Convolution with corner detection kernel")
        kernel_corner = np.array([[1,-2,1], [-2, 4,-2], [1,-2,1]]) 
        op_gray= cv2.filter2D(img_gray,-1,kernel_corner)
        display_image(op_gray)
        if st.button('See the Corner Detection Kernel'):
            st.text(kernel_corner)
        st.markdown("***")
        
def sharpen_kernel():
    st.header('Sharpen Kernel')

    st.subheader('Kernel Design')
    st.write('The sharpern kernel emphasizes the difference in adjacent pixels. The kernel is shown below:')

    st.latex(r'''
    \begin{bmatrix}
0 & -1 & 0\\
-1 &  9 & -1\\
0 & -1 & 0
\end{bmatrix}, 
    ''')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img= np.array(img)

        st.subheader("Original image")
        st.image(img, use_column_width=True,clamp = True)
        img_len1 = np.shape(img)[0]
        img_len2 = np.shape(img)[1]

        #sharpening filter
        st.subheader("Convolution with sharpening filter")
        #sharpening kernel
        sh = st.slider('Change Threshold value',min_value = 1.0,max_value = 5.0, step=0.2) 
        kernel_sharp = np.array([[0,-sh,0], [-sh, 5*sh,-sh], [0,-sh,0]])
        op_sharp = np.zeros((img_len1,img_len2,3), 'uint8')
        op_sharp = do_convolution(img,op_sharp,kernel_sharp)
        display_image(op_sharp)
        if st.button('See the Sharpening Kernel'):
            st.text(kernel_sharp)
        st.markdown("***")
        

if __name__ == "__main__":
    main()