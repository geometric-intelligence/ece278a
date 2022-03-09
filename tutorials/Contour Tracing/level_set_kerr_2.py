import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
from numba import cuda
import numba as nb
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgba2rgb


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)

def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


# RTX 3060 Laptop GPU contain 3840 cuda cores.
@cuda.jit
def gpu_scan_arr(arr, width, height, gpu_result):
    idx = cuda.threadIdx.x
    idy = cuda.blockIdx.x
    
    if(arr[idx][idy]==True):
        #leftTop
        tf = (idx>0 and idy>0 and arr[idx-1][idy-1]==False) or (idx>0 and arr[idx-1][idy]==False) or \
             (idx>0 and idy<width-1 and arr[idx-1][idy+1]==False) or \
             (idy>0 and arr[idx][idy-1]==False) or \
             (idy<width-1 and arr[idx][idy+1]==False) or \
             (idx<height-1 and idy>0 and arr[idx+1][idy-1]==False) or \
             (idx<height-1 and arr[idx+1][idy]==False) or \
             (idx<height-1 and idy<width-1 and arr[idx+1][idy+1]==False)
        if(tf):
            gpu_result[idx][idy] = True
        else:
            gpu_result[idx][idy] = False

            
@cuda.jit
def gpu_expand_contour(arr, width, height, gpu_result):
    idx = cuda.threadIdx.x
    idy = cuda.blockIdx.x
    
    if(True):
        #leftTop
        tf = (idx>0 and idy>0 and arr[idx-1][idy-1]) or (idx>0 and arr[idx-1][idy]) or \
             (idx>0 and idy<width-1 and arr[idx-1][idy+1]) or \
             (idy>0 and arr[idx][idy-1]) or \
             (idy<width-1 and arr[idx][idy+1]) or \
             (idx<height-1 and idy>0 and arr[idx+1][idy-1]) or \
             (idx<height-1 and arr[idx+1][idy]) or \
             (idx<height-1 and idy<width-1 and arr[idx+1][idy+1])
        if(tf):
            gpu_result[idx][idy] = True
        else:
            gpu_result[idx][idy] = False
            

    
@cuda.jit
def gpu_add_contour_color(arr, img, gpu_result):
    idx = cuda.threadIdx.x # threads_per_block
    idy = cuda.blockIdx.x
    if(arr[idx][idy]==True):
        gpu_result[idx][idy][0] = 255.0
        gpu_result[idx][idy][1] = 0.0
        gpu_result[idx][idy][2] = 0.0
    else:
        gpu_result[idx][idy][0] = img[idx][idy][0]
        gpu_result[idx][idy][1] = img[idx][idy][1]
        gpu_result[idx][idy][2] = img[idx][idy][2]

        
class levelSetSolver:
    def __init__(self, dt = 1, sigma = 1, n_iter=100, view3d=False):
        self.sigma = sigma
        self.n_iter = n_iter
        self.dt = dt
        self.view3d = view3d
    
    @nb.jit(forceobj=True)
    def calculate_phi(self):
        if(self.view3d):
            self.phis = np.zeros((self.n_iter, self.phi.shape[0], self.phi.shape[1]))
            for i in range(self.n_iter):
                dphi = np.array(np.gradient(self.phi))
                dphi_norm = np.sqrt(np.sum(np.square(dphi), axis=0))
                dphi_t = self.F * dphi_norm
                self.phi = self.phi + self.dt * dphi_t
                self.phis[i] = self.phi
        else:
            for i in range(self.n_iter):
                dphi = np.array(np.gradient(self.phi))
                dphi_norm = np.sqrt(np.sum(np.square(dphi), axis=0))
                dphi_t = self.F * dphi_norm
                self.phi = self.phi + self.dt * dphi_t
    
    
    def run(self, img):
        # save img
        if(img.dtype=='float'):
            img = (img * 255).round().astype(np.uint8)
        self.img = img
        
        # change into grayscale
        if(len(img.shape)==3):
            img = rgb2gray(img)
        if(img.dtype=='float'):
            img = (img * 255).round().astype(np.uint8)
            
        # start copying code
        img = img - np.mean(img)
        # Smooth the image to reduce noise and separation between noise and edge becomes clear
        img_smooth = scipy.ndimage.filters.gaussian_filter(img, self.sigma)
        self.F = stopping_fun(img_smooth)
        
        self.phi=default_phi(img_smooth)
        self.calculate_phi()
        
    def scan_contour(self):
        self.arr = self.phi > 0.5
        
        arr_device = cuda.to_device(self.arr)         
        gpu_result = cuda.device_array(self.arr.shape)
        
        threads_per_block = self.arr.shape[0]
        blocks_per_grid = self.arr.shape[1]
        gpu_scan_arr[blocks_per_grid, threads_per_block](arr_device, threads_per_block, blocks_per_grid, gpu_result)
        cuda.synchronize()
        
        self.contour = gpu_result.copy_to_host()
        
    def expand_contour(self):
        contour_device = cuda.to_device(self.contour)
        gpu_result = cuda.device_array(self.contour.shape)
        
        threads_per_block = self.contour.shape[0]
        blocks_per_grid = self.contour.shape[1]
        
        gpu_expand_contour[blocks_per_grid, threads_per_block](contour_device, threads_per_block, blocks_per_grid, gpu_result)
        
        cuda.synchronize()
        self.new_contour = gpu_result.copy_to_host()
        
    def write_contour_to_image(self):
        if(len(self.img.shape)<3):# gray
            tempImg = gray2rgb(self.img)
            img_device = cuda.to_device(tempImg)
            gpu_result = cuda.device_array(tempImg.shape)
        else:
            img_device = cuda.to_device(self.img)
            gpu_result = cuda.device_array(self.img.shape)
            
        contour_device = cuda.to_device(self.new_contour)
        
        threads_per_block = self.arr.shape[0]
        blocks_per_grid = self.arr.shape[1]
        
#         print(gpu_result.shape)
        gpu_add_contour_color[blocks_per_grid, threads_per_block](contour_device, img_device, gpu_result)
        cuda.synchronize()
        self.img_with_contour = gpu_result.copy_to_host().astype('uint8')   
