#ECE 278A FINAL PROJECT: ELiminating Haze from Images. Dark Channel Prior Implementation, Canan Cebeci

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#Implementation of a Dark Channel Prior method, proposed by Kaiming He, Jian Sun, Xiaoou Tang

# Get the dark channel of the image
def get_dark_channel (image, patch_size):
    """Get the dark channel prior in the image.
    Parameters
    -----------
    image:  RGB image
    patch_size:  patch size for calculating the dark channel
    Return
    -----------
    The dark channel prior of the image.
    """
    B,G,R = cv2.split(image)
    min_of_min = cv2.min(cv2.min(R,G),B);
    patch = cv2.getStructuringElement(cv2.MORPH_RECT,(patch_size,patch_size))
    dark_channel = cv2.erode(min_of_min,patch)
    return dark_channel
    

# Get the atmospheric light of the image
def get_atmospheric_light(image,dark_channel):
    """Get the atmosphere light in the image.
    Parameters
    -----------
    image:  RGB image
    dark_channel: The dark channel of the image input
    Return
    -----------
    Array of size 3x1 containing the atmospheric light for each RGB channel
    """
    [h,w] = image.shape[:2] #get the image size
    size = h*w
    number_of_pixels = int(max(math.floor(size/1000),1)) #pick the top 0.1% brightest pixels
    dark_vec = dark_channel.reshape(size); #convert the dark  channel value into an array
    image_vec = image.reshape(size,3); #convert the image into an array
    indices = dark_vec.argsort(); # sort the dark channel values to find the brightest
    indices = indices[size-number_of_pixels::] #get the indices of the brighest pixels
    atmospheric_light = np.zeros([1,3]) # initialize the atmospheric light values for each RGB channel
    for i in range(1,number_of_pixels):
        atmospheric_light = atmospheric_light + image_vec[indices[i]] #sum the pixel values of the brightest pixels for each channel
    A = atmospheric_light / number_of_pixels;
    return A
 
 
#Estimate the transmission (this is an implementation of Equation 5)
def get_transmission(image,A,patch_size):
    """Get the transmission estimate.
    Parameters
    -----------
    image:   RGB image
    A:       3x1 array of atmospheric light for each RGB channel
    patch_size:  patch size
    Return
    -----------
    Transmission estimate
    """
    omega = 0.95; # this parameter is for keeping the resulting image natural
    t = np.empty(image.shape,image.dtype); #initialize a matrix to keep transmission estimates
    for i in range(0,3):
        t[:,:,i] = image[:,:,i]/A[0,i] # get the normalized image values for each RGB channel
    transmission = 1 - omega*get_dark_channel(t,patch_size); # equation 5
    return transmission


#Soft matting algorithm to refine the transmission estimate
def soft_matting(image,fltr,r,epsilon):
    """Softmatting algorithm
    Parameters
    -----------
    image:  input RGB image
    fltr:   filter to be guided (the estimated transmission map)
    r:   radius of the guidance
    epsilon: regularizing parameter
    Return
    -----------
    Guided filter.
    """
    mean_image = cv2.boxFilter(image,cv2.CV_64F,(r,r)); #filter the input image with a mean kernel of size rxr
    mean_fltr = cv2.boxFilter(fltr, cv2.CV_64F,(r,r)); #filter the filter with a mean kernel of size rxr
    mean_imagef = cv2.boxFilter(image*fltr,cv2.CV_64F,(r,r)); #filter input image with the filter, and then filter the result with the mean filter
    cov_imagef = mean_imagef - mean_image*mean_fltr; #covariance of the input image and the filter to be guided
    mean_image2 = cv2.boxFilter(image*image,cv2.CV_64F,(r,r));
    variance_image   = mean_image2 - mean_image*mean_image; #find the variance of the input image
    a = cov_imagef/(variance_image + epsilon); #calculate coefficient a (Equation 10)
    b = mean_fltr - a*mean_image; #calculate coefficient b (Equation 11)
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r)); #calculate mean of coefficient a
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r)); #calculate mean of coefficient b
    q = mean_a*image + mean_b; #Equation 12
    return q;


#Calculate the refined transmission matrix using the function soft_matting(image,fltr,r,epsilon)
def refined_transmission(image,transmission_estimate):
    """Calculate refined transmission map
    Parameters
    -----------
    transmission_estimate: estimated transmission map
    Return
    -----------
    Refined transmission matrix.
    """
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); # convert the image into gray scale
    gray_image = np.float64(gray_image)/255;
    r = 60; #radius of the guidance
    epsilon = 0.0001;
    refined = soft_matting(gray_image,transmission_estimate,r,epsilon);
    return refined;

def recover_j(image,transmission_estimate,A,t0):
"""Recover the scene radiance J
Parameters
-----------
image:  input RGB image
transmission_estimate: estimated transmission map
A: atmospheric light
t0: lower bound for keeping some amount of haze in the image

Return
-----------
Scene radiance matrix.
"""
scene_radiance = np.empty(image.shape,image.dtype); #initialize J matrix
transmission_estimate = cv2.max(transmission_estimate,t0);
for i in range(0,3):
    scene_radiance[:,:,i] = (image[:,:,i]-A[0,i])/transmission_estimate + A[0,i] # Equation 13 for each RGB channel
return scene_radiance


#main function
if __name__ == '__main__':

    #First, try dehazing an image with Histogram equalization
    #CLAHE for color images
    or_1 = plt.imread('Images/0018.jpg')
    lab = cv2.cvtColor(or_1, cv2.COLOR_RGB2LAB) # convert rgb to lab format
    lab_RGB = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_RGB[0] = clahe.apply(lab_RGB[0])
    lab_RGB[1] = clahe.apply(lab_RGB[1])
    lab_RGB[2] = clahe.apply(lab_RGB[2])
    lab = cv2.merge(lab_RGB)
    result1 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite('clahe_0018.jpg',cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))

    or_2 = plt.imread('Images/0014.jpg')
    lab = cv2.cvtColor(or_2, cv2.COLOR_RGB2LAB) # convert rgb to lab format
    lab_RGB = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_RGB[0] = clahe.apply(lab_RGB[0])
    lab_RGB[1] = clahe.apply(lab_RGB[1])
    lab_RGB[2] = clahe.apply(lab_RGB[2])
    lab = cv2.merge(lab_RGB)
    result2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    cv2.imwrite('clahe_0014.jpg', cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))

    #plot hazy images and the results
    fig, ax = plt.subplots(ncols=2, nrows =2,figsize=(15, 10))
    ax[0][0].imshow(or_1)
    ax[0][0].set_title("Hazy Image")
    ax[0][0].axis('off')
    ax[0][1].imshow(result1)
    ax[0][1].axis('off')
    ax[0][1].set_title("Result of CLAHE Histogram Equalization")
    ax[1][0].axis('off')
    ax[1][0].imshow(or_2)
    ax[1][0].axis('off')
    ax[1][1].imshow(result2)
    ax[1][1].axis('off')
    plt.show()

    #Now DCP method
    input1 = plt.imread('Images/0018.jpg' );
    image1 = input1.astype('float64')/255;
 
    dark_channel = get_dark_channel(image1,15); #choose patch size to be 15x15
    A = get_atmospheric_light(image1,dark_channel);
    transmission_estimate_first = get_transmission(image1,A,15);
    transmission_estimate_refined1 = refined_transmission(input1,transmission_estimate_first);
    J1 = recover_j(image1,transmission_estimate_refined1,A,0.1); #choose t_0 to be 0.1
    
    input2 = plt.imread('Images/0014.jpg' );
    image2 = input2.astype('float64')/255;
 
    dark_channel = get_dark_channel(image2,15); #choose patch size to be 15x15
    A = get_atmospheric_light(image2,dark_channel);
    transmission_estimate_first = get_transmission(image2,A,15);
    transmission_estimate_refined2= refined_transmission(input2,transmission_estimate_first);
    J2 = recover_j(image2,transmission_estimate_refined2,A,0.1); #choose t_0 to be 0.1
    
    #plot the results
    J1_plot = np.clip(J1, 0, 1)  #convert recovered image into plotable format
    J2_plot = np.clip(J2, 0, 1)  #convert recovered image into plotable format
    fig, ax = plt.subplots(ncols=3,nrows=2, figsize=(25, 10))
    ax[0][0].imshow(input1)
    ax[0][0].set_title("Original Hazy Image")
    ax[0][0].axis('off')
    ax[0][1].imshow(transmission_estimate_refined1, cmap='gray')
    ax[0][1].set_title("Refined Transmission Estimate")
    ax[0][1].axis('off')
    ax[0][2].imshow(J1_plot)
    ax[0][2].set_title("Recovered Image")
    ax[0][2].axis('off')
    ax[1][0].imshow(input2)
    ax[1][0].set_title("Original Hazy Image")
    ax[1][0].axis('off')
    ax[1][1].imshow(transmission_estimate_refined2, cmap='gray')
    ax[1][1].set_title("Refined Transmission Estimate")
    ax[1][1].axis('off')
    ax[1][2].imshow(J2_plot)
    ax[1][2].set_title("Recovered Image")
    ax[1][2].axis('off')
    plt.show()
