import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from skimage import data,io,filters,color,exposure
import imageio
from load_css import local_css
import matplotlib.pyplot as plt
from scipy import ndimage

local_css("style.css")


def main():

    st.sidebar.title('ECE-278A')
    st.sidebar.subheader('Digital Image Processing - W22')
    st.sidebar.markdown('')
    st.sidebar.text('Hello !')
    st.sidebar.markdown('This app was created to understand the basic image processing operations. Select the operator category below to begin.')
    st.sidebar.markdown('')
    selected_box = st.sidebar.selectbox(
    'Operator subcategory',
    ('Pointwise Operators','Neighborhood Operators')
    )
    
    
    if selected_box == 'Pointwise Operators':
        pointwise()
    if selected_box == 'Neighborhood Operators':
        neighborhood()
    
def welcome():
    st.write('# Welcome to this `Streamlit` app !')


def choose_img(user_input):
    '''
    This function is used to choose the image to be used in the app.
    '''
    
    if user_input == 'Mountains':
        file_path = 'img/mountain.jpg'
    elif user_input == 'NYC Bridge':
        file_path = 'img/nyc-bridge.jpg'
    elif user_input == 'Lake Tahoe':
        file_path = 'img/lake-tahoe.jpg'
    elif user_input == 'Busy Street':
        file_path = 'img/busy_street.jpg'
    elif user_input == 'Tarts':
        file_path = 'img/tarts.jpg'
    elif user_input == 'Steam Engine':
        file_path = 'img/engine.jpg'
    elif user_input == 'Cake':
        file_path = 'img/cake.jpg'
    elif user_input == 'Bike Rack':
        file_path = 'img/bike-rack.jpeg'
    elif user_input == 'Trees':
        file_path = 'img/trees.jpeg'
    elif user_input == 'Waves':
        file_path = 'img/waves.jpeg'
    elif user_input == 'Branch':
        file_path = 'img/branch.jpeg'
    return file_path



def pointwise():
    st.write('# Pointwise Operators')

    t = """<div class=TextBox> Pointwise operators apply the same operation to every individual pixel. The result
    of the computation is independent of the pixel location and value of neighboring pixel values.
    In the equations below <span class=math-text>f(.)</span>, <span class=math-text> g(.)</span> represent the original and resultant image respectively.
    <span class=math-text>h(.)</span> represents the function corresponding to the operator.
    <br>
    For continuous images, this can be denoted by: 
    </div>
    """
    st.markdown(t, unsafe_allow_html=True)

    st.latex("g(\mathbf{x}) = h(f(\mathbf{x}))")

    t = """<div class=TextBox> For discrete (sampled) images:</div>
    """
    st.markdown(t, unsafe_allow_html=True)
    
    st.latex("g(i,j) = h(f(i,j))\:\:\: {where\:} \mathbf{x}=(i,j)")

    t = "## <span class=blue_text>1. Multiplicative Gain and Bias</span>"
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=TextBox>This point operation involves multiplication and addition with a constant. Mathematically
    for a discrete image: </div>
    """
    st.markdown(t, unsafe_allow_html=True)
    
    st.latex("g(i,j) = a\:f(i,j) + b")
    
    t = """<div class=TextBox>The parameters <span class=math-text>a</span>(>0) and <span class=math-text>b</span>
    are called the <i>gain</i> and <i>bias</i> respectively. While <span class=math-text>a</span> controls the image contrast, <span class=math-text>b</span>
    determines the brightness of the image.</div><br>
    """
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)

    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'))

    if test_option == 'No':
        img_choice = st.selectbox('Select an image',('Mountains','NYC Bridge', 'Lake Tahoe', 'Busy Street', 'Tarts'))
        
        file_path = choose_img(img_choice)
        im = Image.open(file_path)
        im = np.array(im)/255

        contrast = st.slider('Contrast (Gain)',min_value=0.0,max_value=2.0, step=0.01, value=1.0)
        brightness = st.slider('Brightness (Bias)',min_value=0.0,max_value=2.0, step=0.01, value=0.0)
        modified_img = contrast*im + brightness
        st.image(modified_img,use_column_width=True, clamp=True)
    else:
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'],key="img")
        if image_file is not None:
            im = Image.open(image_file)
            im = np.array(im)/255
            contrast = st.slider('Contrast (Gain)',min_value=0.0,max_value=2.0, step=0.01)
            brightness = st.slider('Brightness (Bias)',min_value=0.0,max_value=2.0, step=0.01)
            modified_img = contrast*im + brightness
            st.image(modified_img,use_column_width=True, clamp=True)

    st.text("")
    st.text("")
    # --------------------------------------------------------------
    st.text("")
    st.text("")




    t = "## <span class=blue_text>2. Linear blend operator</span>"
    st.markdown(t, unsafe_allow_html=True)
    
    t = """<div class=TextBox>This is a two input (<i>dyadic</i>) operator which performs
    a smooth cross-dissolve between two images as the value of the parameter α is changed from 
    0 to 1. Mathematically, let <span class=math-text>f_0(.)</span> and
    <span class=math-text>f_1(.)</span> be two images, then the resultant image 
    <span class=math-text>g(.)</span> is given by: </div>"""

    st.markdown(t, unsafe_allow_html=True)
    st.latex("g(i,j) = (1-α)f_0(i,j) + α\:f_1(i,j)")

    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)
    
    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'), key='linear-blend')

    if test_option == 'No':
        puppy_img = Image.open('img/puppy.jpg')
        puppy_img = np.array(puppy_img)/255

        cat_img = Image.open('img/cat.jpg').resize((puppy_img.shape[1], puppy_img.shape[0]))
        cat_img = np.array(cat_img)/255

        alpha = st.slider('Parameter α',min_value=0.0,max_value=1.0, step=0.01)
        blend_img = alpha*cat_img + (1-alpha)*puppy_img
        st.image(blend_img,use_column_width=True)

    else:
        image_file1 = st.file_uploader('Upload 1st image', type=['jpg', 'jpeg', 'png'],key="linear-blend-1")
        image_file2 = st.file_uploader('Upload 2nd image', type=['jpg', 'jpeg', 'png'],key="linear-blend-2")
        if image_file1 is not None and image_file2 is not None:
            im1 = Image.open(image_file1)
            im1 = np.array(im1)/255
            im2 = Image.open(image_file2).resize((im1.shape[1], im1.shape[0]))
            im2 = np.array(im2)/255
            alpha = st.slider('Parameter α',min_value=0.0,max_value=1.0, step=0.01)
            blend_img = alpha*im1 + (1-alpha)*im2
            st.image(blend_img,use_column_width=True)

    st.text("")
    st.text("")
    # --------------------------------------------------------------
    st.text("")
    st.text("")


    t = "## <span class=blue_text>3. Histogram Equalization</span>"
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=TextBox> This method utilizes an image's histogram to adjust the contrast. An image histogram provides a visual representation of the number of pixels in an image as a function of their intensity. This is accomplished by
    spreading out the most frequent intensity values in the image, resulting in a more uniform distribution of pixel intensity across the image. To do this, we can replace every intensity value found in the image with its corresponding CDF value. The formula can be seen below, where p is any discrete probability distribution:
    </div>"""
    
    st.markdown(t, unsafe_allow_html=True)

    st.latex("CDF(\mathbf{a}) = \sum_{b=0}^{a}p(\mathbf{b}))")

    st.text("")
    st.text("")


    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)

    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'), key='histogram-equalization')

    if test_option == 'No':
        img_choice = st.selectbox('Select an image',('Branch','Waves','Trees'))
        file_path = choose_img(img_choice)
        low_contrast = Image.open(file_path)
        low_contrast = np.array(low_contrast)
        low_contrast_gray = color.rgb2gray(low_contrast)*255

        # CDF/hist for low contrast img
        fig1 = plt.figure()
        hist,bins = np.histogram(low_contrast_gray.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(low_contrast_gray.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('CDF','Histogram'), loc = 'upper left')

        # hist equalized img
        hist_eq_im = exposure.equalize_hist(low_contrast_gray)*255

        # CDF/hist for hist equalized img
        fig2 = plt.figure()
        hist,bins = np.histogram(hist_eq_im.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(hist_eq_im.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('CDF','Histogram'), loc = 'upper left')


        # layout formatting
        col1, col2 = st.columns(2)
        with col1:
            st.image(low_contrast,clamp=True)
            t = """<div class=plain-text>Low Contrast</div>"""
            st.markdown(t, unsafe_allow_html=True)
            st.pyplot(fig1)
        with col2:
            st.image(hist_eq_im/255,clamp=True)
            t = """<div class=plain-text>Histogram Equalized</div>"""
            st.markdown(t, unsafe_allow_html=True)
            st.pyplot(fig2)
    else:
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'],key="hist_eq")
        if image_file is not None:
            im = Image.open(image_file)
            low_contrast = np.array(im)
            low_contrast_gray = color.rgb2gray(low_contrast)*255

            # CDF/hist for low contrast img
            fig1 = plt.figure()
            hist,bins = np.histogram(low_contrast_gray.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(low_contrast_gray.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('CDF','Histogram'), loc = 'upper left')

            # hist equalized img
            hist_eq_im = exposure.equalize_hist(low_contrast_gray)*255

            # CDF/hist for hist equalized img
            fig2 = plt.figure()
            hist,bins = np.histogram(hist_eq_im.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            plt.plot(cdf_normalized, color = 'b')
            plt.hist(hist_eq_im.flatten(),256,[0,256], color = 'r')
            plt.xlim([0,256])
            plt.legend(('CDF','Histogram'), loc = 'upper left')


            # layout formatting
            col1, col2 = st.columns(2)
            with col1:
                st.image(low_contrast,clamp=True)
                t = """<div class=plain-text>Low Contrast</div>"""
                st.markdown(t, unsafe_allow_html=True)
                st.pyplot(fig1)
            with col2:
                st.image(hist_eq_im/255,clamp=True)
                t = """<div class=plain-text>Histogram Equalized</div>"""
                st.markdown(t, unsafe_allow_html=True)
                st.pyplot(fig2)

    st.text("")
    st.text("")






def neighborhood():
    st.write('# Neighborhood Operators')
   
    t = """<div class=TextBox> Neighborhood operators use the pixels in the vicinity of a given pixel to 
    compute the result. These operators are generally used to emphasize edges, soft blur, add noise, etc.
    For example, the output of the correlation operator is the weighted sum of pixel values in a small neighborhood
    of a given pixel, defined by a kernel (matrix of weights). Mathematically: </div>"""
    
    st.markdown(t, unsafe_allow_html=True)

    st.latex("g(i,j) = \sum_{k,l}f(i+k,j+l) \:h(k,l)")

    t = """<div class=TextBox> The <i>kernel</i> <span class=math-text>h(k,l)</span> is a matrix of weights also called the <i>filter coefficients</i>."""
    st.markdown(t, unsafe_allow_html=True)


    t = "## <span class=blue_text>1. Sobel operator</span>"
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=TextBox> The Sobel operator is a two-dimensional operator that detects edges in an image by computing
    computing the gradient. It uses two 3 X 3 kernels (one each for horizontal-Gx and vertical-Gy) which are convolved to approximate 
    the derivatives at each pixel.</div>"""

    st.markdown(t, unsafe_allow_html=True)

    image = Image.open('img/sobel_kernel.png')
    kernel = (np.array(image)/255)
    st.image(kernel,  width=700, clamp=True)

    st.text("")
    

    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)

    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'), key='sobel')
    
    if test_option == 'No':
        img_choice = st.selectbox('Select an image',('Steam Engine','NYC Bridge', 'Cake', 'Busy Street', 'Bike Rack'))
        
        file_path = choose_img(img_choice)
        im = Image.open(file_path)
        im = np.array(im)/255
        
        if len(im.shape)==3:
            im = color.rgb2gray(im)
        
        edge_im = filters.sobel(im)
        
        st.image(edge_im,use_column_width=True, clamp=True)
    else:
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'],key="sobel")
        if image_file is not None:
            im = Image.open(image_file)
            im = np.array(im)/255
            
            if len(im.shape)==3:
                im = color.rgb2gray(im)
            
            edge_im = filters.sobel(im)
            st.image(edge_im,use_column_width=True, clamp=True)


    st.text("")
    st.text("")
    # --------------------------------------------------------------
    st.text("")
    st.text("")

    t = "## <span class=blue_text>2. Gaussian blur</span>"
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=TextBox>The Gaussian filter is a low-pass blurring filter. It smoothes the color transition
    in the image from one side of the edge to the another rather than making it sudden. <br>
    The weights of the kernel are such that the pixels nearest to the center of the kernel are given more weight than 
    those far away from the center. An example of a 3 X 3 kernel is shown below:</div>"""

    st.markdown(t, unsafe_allow_html=True)

    image = Image.open('img/gaussian_kernel.png')
    kernel = (np.array(image)/255)
    st.image(kernel,  width=700, clamp=True)


    st.text("")
    st.text("")
    
    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)

    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'), key='gaussian-blur')

    if test_option == 'No':
        img_choice1 = st.selectbox('Select an image',('Steam Engine','NYC Bridge', 'Cake', 'Busy Street', 'Bike Rack'), key="gaussian_blur")
        
        file_path = choose_img(img_choice1)
        im = Image.open(file_path)

        converted_img = np.array(im.convert('RGB'))
        var = st.slider('Standard Deviation',0,20,step=1)
        #slider = st.sidebar.slider('Adjust the intensity', 5, 21, 5, step=2)
        #gray_img = color.rgb2gray(converted_img)
        blur_image = filters.gaussian(converted_img,sigma=var,channel_axis=2)
        st.image(blur_image,use_column_width=True,clamp=True)


        pixel_l1 = converted_img[:400,:400,:]
        pixel_l2 = blur_image[:400,:400,:]



        t = """<div class=underline-text>Upper Sectional View</div>"""
        st.markdown(t, unsafe_allow_html=True)
        st.text("")
        col1, col2 = st.columns(2)

        with col1:
            st.image(pixel_l1, clamp=True)
            t = """<div class=plain-text>Before blur</div>"""
            st.markdown(t, unsafe_allow_html=True)

        with col2:
            st.image(pixel_l2,clamp=True)
            t = """<div class=plain-text>After blur</div>"""
            st.markdown(t, unsafe_allow_html=True)
        
    else:
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'],key="gaussian-blur")
        if image_file is not None:
            im = Image.open(image_file)
            converted_img = np.array(im.convert('RGB'))
            var = st.slider('Standard Deviation',0,20,step=1)
            #slider = st.sidebar.slider('Adjust the intensity', 5, 21, 5, step=2)
            #gray_img = color.rgb2gray(converted_img)
            blur_image = filters.gaussian(converted_img,sigma=var,channel_axis=2)
            st.image(blur_image,use_column_width=True,clamp=True)



            pixel_l1 = converted_img[:400,:400,:]
            pixel_l2 = blur_image[:400,:400,:]



            t = """<div class=underline-text>Upper Sectional View</div>"""
            st.markdown(t, unsafe_allow_html=True)
            st.text("")
            col1, col2 = st.columns(2)

            with col1:
                st.image(pixel_l1, clamp=True)
                t = """<div class=plain-text>Before blur</div>"""
                st.markdown(t, unsafe_allow_html=True)

            with col2:
                st.image(pixel_l2,clamp=True)
                t = """<div class=plain-text>After blur</div>"""
                st.markdown(t, unsafe_allow_html=True)
    
    st.text("")
    st.text("")
    st.text("")

    # --------------------------------------------------------------

    t = "## <span class=blue_text>3. Moving Average filter</span>"
    st.markdown(t, unsafe_allow_html=True)

    t = """<div class=TextBox>The Moving Average filter is a low-pass blurring filter. It gradually blurs the image so that harsh transitions appear smoother. <br>
    It averages the pixel values in a K x K by window, so basically performs convolution with a normalized constant kernel. The structure of a K x K kernel is shown below:</div>"""

    st.markdown(t, unsafe_allow_html=True)

    image = Image.open('img/moving_avg.png')
    kernel = (np.array(image)/255)
    st.image(kernel, width=700, clamp=True)


    st.text("")
    st.text("")
    
    t = """<div class=demonstration>Demonstration</div>"""
    st.markdown(t, unsafe_allow_html=True)

    test_option = st.selectbox('Do you want to upload an image ?',('No','Yes'), key='moving-avg')

    if test_option == 'No':
        img_choice1 = st.selectbox('Select an image',('Steam Engine','NYC Bridge', 'Cake', 'Busy Street', 'Bike Rack'), key="moving-avg")
        
        file_path = choose_img(img_choice1)
        im = Image.open(file_path)

        converted_img = np.array(im.convert('RGB'))
        dims = st.slider('Kernel Size',1,20,step=1)
        #slider = st.sidebar.slider('Adjust the intensity', 5, 21, 5, step=2)
        #gray_img = color.rgb2gray(converted_img)
        #blur_image = cv2.boxFilter(converted_img,-1,(dims,dims),cv2.BORDER_DEFAULT)
        #st.image(blur_image,use_column_width=True,clamp=True)
        

        # Convolving with the appropriate kernel for each channel
        kernel = np.ones((dims,dims),np.float32)/(dims*dims)
        blur_image_r = ndimage.convolve(converted_img[:,:,0], kernel, mode='constant', cval=0.0)
        blur_image_g = ndimage.convolve(converted_img[:,:,1], kernel, mode='constant', cval=0.0)
        blur_image_b = ndimage.convolve(converted_img[:,:,2], kernel, mode='constant', cval=0.0)
        
        blur_image = np.stack((blur_image_r,blur_image_g,blur_image_b),axis=2)
        st.image(blur_image,use_column_width=True,clamp=True)


        pixel_l1 = converted_img[:400,:400,:]
        pixel_l2 = blur_image[:400,:400,:]



        t = """<div class=underline-text>Upper Sectional View</div>"""
        st.markdown(t, unsafe_allow_html=True)
        st.text("")
        col1, col2 = st.columns(2)

        with col1:
            st.image(pixel_l1, clamp=True)
            t = """<div class=plain-text>Before blur</div>"""
            st.markdown(t, unsafe_allow_html=True)

        with col2:
            st.image(pixel_l2,clamp=True)
            t = """<div class=plain-text>After blur</div>"""
            st.markdown(t, unsafe_allow_html=True)
        
    else:
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'],key="moving-avg")
        if image_file is not None:
            im = Image.open(image_file)
            converted_img = np.array(im.convert('RGB'))
            dims = st.slider('Kernel Size',1,20,step=1)
            #slider = st.sidebar.slider('Adjust the intensity', 5, 21, 5, step=2)
            #gray_img = color.rgb2gray(converted_img)
            #blur_image = cv2.boxFilter(converted_img,-1,(dims,dims),cv2.BORDER_DEFAULT)
            #st.image(blur_image,use_column_width=True,clamp=True)


            # Convolving with the appropriate kernel for each channel
            kernel = np.ones((dims,dims),np.float32)/(dims*dims)
            blur_image_r = ndimage.convolve(converted_img[:,:,0], kernel, mode='constant', cval=0.0)
            blur_image_g = ndimage.convolve(converted_img[:,:,1], kernel, mode='constant', cval=0.0)
            blur_image_b = ndimage.convolve(converted_img[:,:,2], kernel, mode='constant', cval=0.0)
        
            blur_image = np.stack((blur_image_r,blur_image_g,blur_image_b),axis=2)
            st.image(blur_image,use_column_width=True,clamp=True)



            pixel_l1 = converted_img[:400,:400,:]
            pixel_l2 = blur_image[:400,:400,:]



            t = """<div class=underline-text>Upper Sectional View</div>"""
            st.markdown(t, unsafe_allow_html=True)
            st.text("")
            col1, col2 = st.columns(2)

            with col1:
                st.image(pixel_l1, clamp=True)
                t = """<div class=plain-text>Before blur</div>"""
                st.markdown(t, unsafe_allow_html=True)

            with col2:
                st.image(pixel_l2,clamp=True)
                t = """<div class=plain-text>After blur</div>"""
                st.markdown(t, unsafe_allow_html=True)
    





if __name__=="__main__":
    main()


