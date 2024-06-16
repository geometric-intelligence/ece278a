import numpy as np

def box_filter(img, r):
    """Apply a box filter to the image."""
    kernel_size = 2 * r + 1
    return np.cumsum(np.cumsum(img, axis=0), axis=1)[kernel_size - 1:, kernel_size - 1:] - \
           np.cumsum(np.cumsum(img, axis=0), axis=1)[kernel_size - 1:, :-kernel_size] - \
           np.cumsum(np.cumsum(img, axis=0), axis=1)[:-kernel_size, kernel_size - 1:] + \
           np.cumsum(np.cumsum(img, axis=0), axis=1)[:-kernel_size, :-kernel_size]

def guided_filter(I, p, r, eps):
    """Perform guided filtering."""
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    
    # Box filter
    N = box_filter(np.ones(I.shape), r)

    # Means
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(I * p, r) / N
    mean_II = box_filter(I * I, r) / N
    
    # Covariances and variances
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    
    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Means of a and b
    mean_a = box_filter(a, r) / N
    mean_b = box_filter(b, r) / N
    
    # Output image
    q = mean_a * I + mean_b
    return q

# Handle edge cases for the box_filter function
def box_filter(img, r):
    """Apply a box filter to the image with padding."""
    padded_img = np.pad(img, ((r, r), (r, r)), 'constant', constant_values=0)
    return np.cumsum(np.cumsum(padded_img, axis=0), axis=1)[2 * r:, 2 * r:] - \
           np.cumsum(np.cumsum(padded_img, axis=0), axis=1)[2 * r:, :-2 * r] - \
           np.cumsum(np.cumsum(padded_img, axis=0), axis=1)[:-2 * r, 2 * r:] + \
           np.cumsum(np.cumsum(padded_img, axis=0), axis=1)[:-2 * r, :-2 * r]
