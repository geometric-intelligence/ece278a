
import numpy as np
import cv2


def guidedFilter(img, p, r=40, e=1e-3):
    '''
    Guided Filter Implementation
    '''
    H, W, C = img.shape 
    #S is a matrix with the sizes of each local patch (window wk)
    S = boxfilter(np.ones((H, W)), r)
    
    #the mean value of each channel in img
    mean_i = np.zeros((C, H, W))
    
    for c in range(0, C):
        mean_i[c] = boxfilter(img[:,:, c], r)/S
    
    #the mean of the guided filter p
    mean_p = boxfilter(p, r)/S
    
    mean_ip = np.zeros((C, H, W))
    for c in range(0, C):
        mean_ip[c] = boxfilter(img[:,:,c]*p, r)/S
    
    cov_ip = np.zeros((C, H, W))
    for c in range(0, C):
        cov_ip[c] = mean_ip[c] - mean_i[c]*mean_p
    

    #The variance in each window is a 3x3 symmetric matrix with variance as its values:
    #           rr, rg, rb
    #   sigma = rg, gg, gb
    #           rb, gb, bb
    var_i = np.zeros((C, C, H, W))
    #variance of (R, R)
    var_i[0, 0] = boxfilter(img[:,:,0]*img[:,:,0], r)/S - mean_i[0]*mean_i[0]
    #variance of (R, G)
    var_i[0, 1] = boxfilter(img[:,:,0]*img[:,:,1], r)/S - mean_i[0]*mean_i[1]
    #variance of (R, B)
    var_i[0, 2] = boxfilter(img[:,:,0]*img[:,:,2], r)/S - mean_i[0]*mean_i[2]
    #variance of (G, G)
    var_i[1, 1] = boxfilter(img[:,:,1]*img[:,:,1], r)/S - mean_i[1]*mean_i[1]
    #variance of (G, B)
    var_i[1, 2] = boxfilter(img[:,:,1]*img[:,:,2], r)/S - mean_i[1]*mean_i[2]
    #variance of (B, B)
    var_i[2, 2] = boxfilter(img[:,:,2]*img[:,:,2], r)/S - mean_i[2]*mean_i[2]
    
    a=np.zeros((H,W,C))
    
    for i in range(0, H):
        for j in range(0, W):
            sigma = np.array([ [var_i[0, 0, i, j], var_i[0, 1, i, j], var_i[0, 2, i, j]], 
                                  [var_i[0, 1, i, j], var_i[1, 1, i, j], var_i[1, 2, i, j]],
                                  [var_i[0, 2, i, j], var_i[1, 2, i, j], var_i[2, 2, i, j]]])
            

            cov_ip_ij = np.array([ cov_ip[0, i, j], cov_ip[1, i, j], cov_ip[2, i, j]])
            
            a[i, j] = np.dot(cov_ip_ij, np.linalg.inv(sigma + e*np.identity(3)))
    
    b = mean_p - a[:,:,0]*mean_i[0,:,:] - a[:,:,1]*mean_i[1,:,:] - a[:,:,2]*mean_i[2,:,:] 


    pp = ( boxfilter(a[:,:,0], r)*img[:,:,0]
          +boxfilter(a[:,:,1], r)*img[:,:,1]
          +boxfilter(a[:,:,2], r)*img[:,:,2]
          +boxfilter(b, r) )/S
    
    return pp


def boxfilter(m, r):
    # Reference: https://fukushima.web.nitech.ac.jp/paper/2017_iwait_nakamura.pdf
    """
    Fast box filtering implementation, O(1) time.
    """
    h, w = m.shape
    output = np.zeros(m.shape) 
    
    #cumulative sum over y axis
    ysum = np.cumsum(m, axis=0) 
    output[0:r+1, : ] = ysum[r:(2*r)+1, : ]
    output[r+1:h-r, : ] = ysum[(2*r)+1: , : ] - ysum[ :h-(2*r)-1, : ]
    output[(-r): , : ] = np.tile(ysum[-1, : ], (r, 1)) - ysum[h-(2*r)-1:h-r-1, : ]

    #cumulative sum over x axis
    xsum = np.cumsum(output, axis=1)
    output[ : , 0:r+1] = xsum[ : , r:(2*r)+1]
    output[ : , r+1:w-r] = xsum[ : , (2*r)+1: ] - xsum[:, :w-(2*r)-1]
    output[ : , -r: ] = np.tile(xsum[ : , -1][:, None], (1, r)) - xsum[ : , w-(2*r)-1:w-r-1]

    return output


def darkChannel(img, ps=15):
    '''
    Calculate Dark Channel of an RGB Image
    '''
    half_width = int(ps/2)
    impad = np.pad(img, [(half_width,half_width), (half_width,half_width) , (0,0)], 'edge')
    
    dark = np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(half_width, (img.shape[0]+half_width)):
        for j in range(half_width, (img.shape[1]+half_width)):
            patch = impad[i-half_width:i+1+half_width, j-half_width:j+1+half_width]
            dark[i-half_width, j-half_width] = np.min(patch)
    
    return dark


def transmission(img, A, w=0.95):
  '''
  Estimation of Atmospheric Transmission Map
  '''
  nimg = np.zeros(img.shape)
    
  for c in range(0, img.shape[2]):
      nimg[:,:,c] = img[:,:,c]/A[c]
    
  # estimate the dark channel of the normalized haze image
  dark = darkChannel(nimg)
    
  # calculates the transmisson t
  t = 1-w*dark+0.1
    
  return t


def atmLight(img, dark, px=1e-3):
    '''
    Estimation of Global Atmospheric Light
    '''
    imgavec = np.resize(img, (img.shape[0]*img.shape[1], img.shape[2]))
    jdarkvec = np.reshape(dark, dark.size)
    
    numpx = int(dark.size * px)
    
    isjd = np.argsort(-jdarkvec)

    asum = np.zeros((3), dtype=np.float32)
    for i in range(0, numpx):
        asum[:] += imgavec[isjd[i], :]
  
    A = np.zeros((3), dtype=np.float32)
    A[:] = asum[:]/numpx
  
    return A


def recover(img, A, t, tmin=0.1):
    """
    Recover dehazed image using estimations of transmission map and global atmospheric light
    """
    j = np.zeros(img.shape)
    
    for c in range(img.shape[2]):
        j[:,:, c] = ((img[:,:,c] - A[c]) / np.maximum(t[:,:], tmin)) + A[c]
    
    return j/np.max(j) # scale between 0 and 1
