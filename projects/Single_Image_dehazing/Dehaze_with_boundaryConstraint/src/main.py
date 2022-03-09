
import cv2
from matplotlib.pyplot import plot
import numpy as np

from Airlight import Airlight
from Estimate_depth import Estimate_lowerbound
from estimate_t import estimate_t_x
from removeHaze import removeHaze
import os
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':

    #Single Image
    # num_files = 1;
    # HazeImg_dir = '/home/abhi/ucsb/img_comp/final/Images'
    # Clear_Img_dir = '/home/abhi/ucsb/img_comp/final/Images'
    # haze_file = 'bridge.jpg'
    # clear_file = 'bridge.jpg'

    #Outdoor
    haze_dir_path = './SOTS/outdoor/hazy'
    gt_dir_path = './SOTS/outdoor/gt'

    #Indoor
    # haze_dir_path = './SOTS/indoor/hazy'
    # gt_dir_path = './SOTS/indoor/gt'
    # dir_list = os.listdir(haze_dir_path)
    # dir_list = [s for s in dir_list if "_10" in s]
    # HazeImg_dir = sorted(dir_list)

    #Synthetic
    # haze_dir_path = './SOTS/synthetic/hazy'
    # gt_dir_path = './SOTS/synthetic/gt'

    
    # Uncomment this for outdoor and synthetic
    HazeImg_dir = sorted(os.listdir(haze_dir_path))
    Clear_Img_dir = sorted(os.listdir(gt_dir_path))
    
    num_files = 50 # Number of images to run

    error_arr = []
    mse_arr = []
    mse_arr1 = []
    ssim_arr = []
    ssim_arr1 = []

    for i in range(num_files):
        haze_file = HazeImg_dir[i] ##Uncomment for batch run
        HazeImg_path = os.path.join( './SOTS/outdoor/hazy', haze_file) #Change path for different datasets

        clear_file = Clear_Img_dir[i] ##Uncomment for batch run
        ClearImg_path = os.path.join('./SOTS/outdoor/gt', clear_file) #Change path for different datasets

        

        HazeImg = cv2.imread(HazeImg_path)
        ClearImg = cv2.imread(ClearImg_path)
        # Resize image
        '''
        Channels = cv2.split(HazeImg)
        rows, cols = Channels[0].shape
        HazeImg = cv2.resize(HazeImg, (int(0.4 * cols), int(0.4 * rows)))
        '''

        # Estimate Airlight
        windowSze = 5
        A = Airlight(HazeImg, windowSze)

        # Calculate Boundary Constraints
        windowSze = 15
        C0 = 20         # Default value = 20 (as recommended in the paper)
        C1 = 300        # Default value = 300 (as recommended in the paper)
        Transmission = Estimate_lowerbound(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

        # Refine estimate of transmission
        regularize_lambda = 0.5       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
        sigma = 0.5
        Transmission = estimate_t_x(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

        # Perform DeHazing
        HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

        mse = np.zeros((2,1))
        ssim_err= np.zeros((2,1))
        mse[0] = np.sqrt(np.mean(np.square(np.subtract(HazeImg,ClearImg ))))
        mse[1] = np.sqrt(np.mean(np.square(np.subtract(HazeCorrectedImg, ClearImg))))
        mse_arr.append(mse[0])
        mse_arr1.append(mse[1])
        #print('mean square error on original Dehaze image ', mse[0])
        #print('mean square error on modified Dehaze image ', mse[1])

        haze_image_gray = cv2.cvtColor(HazeImg, cv2.COLOR_BGR2GRAY)
        clear_image_gray = cv2.cvtColor(ClearImg, cv2.COLOR_BGR2GRAY)
        corrected_image_gray = cv2.cvtColor(HazeCorrectedImg, cv2.COLOR_BGR2GRAY)

        ssim_err[0] = ssim(haze_image_gray, clear_image_gray)
        ssim_err[1] = ssim(corrected_image_gray, clear_image_gray)
        ssim_arr.append(ssim_err[0])
        ssim_arr1.append(ssim_err[1])
        #print(str(ssim_err[0]))
        #print(str(ssim_err[1]))

        # Display images 
        # cv2.imshow('Original', ClearImg)
        # cv2.imshow('Hazy',HazeImg)
        # cv2.imshow('Result', HazeCorrectedImg)
        # cv2.waitKey(0)


    print("Avg SSIM  of hazy images" + str(np.average(ssim_arr)))
    print("Avg SSIM of corrected images" + str(np.average(ssim_arr1)))

    print("Avg MSE of hazy images" + str(np.average(mse_arr)))
    print("Avg MSE of corrected images" + str(np.average(mse_arr1)))


    cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)
