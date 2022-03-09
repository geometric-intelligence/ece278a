from pickletools import uint8
import cv2;
import math;
import numpy as np;
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib.pyplot import plot

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    
    #Outdoor
    # haze_dir_path = './SOTS/outdoor/hazy'
    # gt_dir_path = './SOTS/outdoor/gt'
    #Indoor
    # haze_dir_path = './SOTS/indoor/hazy'
    # gt_dir_path = './SOTS/indoor/gt'
    # dir_list = os.listdir(haze_dir_path)
    # dir_list = [s for s in dir_list if "_10" in s]
    # HazeImg_dir = sorted(dir_list)
    #Synthetic
    haze_dir_path = './SOTS/synthetic/hazy'
    gt_dir_path = './SOTS/synthetic/gt'

    # Uncomment this for outdoor and synthetic
    HazeImg_dir = sorted(os.listdir(haze_dir_path))

    Clear_Img_dir = sorted(os.listdir(gt_dir_path))
    num_files = len(HazeImg_dir)#50 # Number of images to run
    error_arr = []
    mse_arr = []
    mse_arr1 = []
    ssim_arr = []
    ssim_arr1 = []
    for i in range(num_files):
        haze_file = HazeImg_dir[i]
        HazeImg_path = os.path.join('./SOTS/synthetic/hazy', haze_file) #Change path for different datasets
        clear_file = Clear_Img_dir[i]
        ClearImg_path = os.path.join('./SOTS/synthetic/gt', clear_file) #Change path for different datasets


        HazeImg = cv2.imread(HazeImg_path)
        
        ClearImg = cv2.imread(ClearImg_path)
        # Resize image
        '''
        Channels = cv2.split(HazeImg)
        rows, cols = Channels[0].shape
        HazeImg = cv2.resize(HazeImg, (int(0.4 * cols), int(0.4 * rows)))
        '''
        # dark = DarkChannel(I,15);
        # A = AtmLight(I,dark);
        # te = TransmissionEstimate(I,A,15);
        # t = TransmissionRefine(HazeImg,te);
        # HazeCorrectedImg = Recover(I,t,A,0.1);
        
        lab = cv2.cvtColor(HazeImg, cv2.COLOR_RGB2LAB) # convert rgb to lab format
        lab_divide = cv2.split(lab)
        lab_divide =list(lab_divide)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        data = clahe.apply(lab_divide[0])
        lab_divide[0] = list(data);
        lab_divide = tuple(lab_divide)
        lab = cv2.merge(lab_divide)
        result1 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        HazeCorrectedImg = cv2.imwrite('clahes/clahe_0022.jpg',cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))

        HazeCorrectedImg = HazeCorrectedImg * 255;
        HazeCorrectedImg = HazeCorrectedImg.astype(int)
        HazeCorrectedImg=  np.array(HazeCorrectedImg, dtype=np.uint8);
     
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
        corrected_image_gray = cv2.cvtColor(HazeCorrectedImg , cv2.COLOR_BGR2GRAY)
        ssim_err[0] = ssim(haze_image_gray, clear_image_gray)
        ssim_err[1] = ssim(corrected_image_gray, clear_image_gray)
        ssim_arr.append(ssim_err[0])
        ssim_arr1.append(ssim_err[1])
        #print(str(ssim_err[0]))
        #print(str(ssim_err[1]))
        # Display images
        cv2.imshow('Original', ClearImg)
        cv2.imshow('Hazy',HazeImg)
        cv2.imshow('Result', HazeCorrectedImg)
        cv2.waitKey(1500)
    print("Avg SSIM  of hazy images" + str(np.average(ssim_arr)))
    print("Avg SSIM of corrected images" + str(np.average(ssim_arr1)))
    print("Avg MSE of hazy images" + str(np.average(mse_arr)))
    print("Avg MSE of corrected images" + str(np.average(mse_arr1)))
    #cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)











