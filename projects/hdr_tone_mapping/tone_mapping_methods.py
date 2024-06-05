import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from fractions import Fraction
import guided_filter as gf

def log_lo_median(pixels, gamma):
    pixels_gamma = np.power(pixels, gamma)
    pixels_gamma_f = pixels_gamma[pixels_gamma <= 0.5]
    return np.median(pixels_gamma_f) if pixels_gamma_f.size > 0 else 0

def log_hi_median(pixels, gamma):
    pixels_gamma = np.power(pixels, gamma)
    pixels_gamma_f = pixels_gamma[pixels_gamma > 0.5]
    return np.median(pixels_gamma_f) if pixels_gamma_f.size > 0 else 0

def display_gamma_curves(gamma_lo, gamma_hi):
    
    gamma_values = [gamma_lo, gamma_hi]

    # Generate input values from 0 to 1
    input_values = np.linspace(0, 1, 256)

    # Plot gamma curves for each gamma value
    plt.figure(figsize=(9, 6))
    for gamma in gamma_values:
        output_values = input_values ** gamma
        plt.plot(input_values, output_values, label=f'Gamma = {gamma}')

    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.title('Gamma Correction Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def adaptive_dynamic_range_adjustment(img, display_gamma=False):

    # Logarithmic Normalization
    luminance = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    log_base = np.max(luminance) + 1
    luminance_log = np.emath.logn(log_base, luminance + 1) 

    # Adaptive Gamma Correction
    gamma_threshold = 0.5
    luminance_log_lo = luminance_log[luminance_log <= gamma_threshold] 
    luminance_log_hi = luminance_log[luminance_log > gamma_threshold]

    # Standard deviations of dark and bright pixels
    std_lo = np.std(luminance_log_lo) if luminance_log_lo.size > 0 else 0
    std_hi = np.std(luminance_log_hi) if luminance_log_hi.size > 0 else 0

    # Expected median values
    median_lo = float(Fraction(1, 3)) + std_lo
    median_hi = 1.0 - std_hi

    gamma_lo_range = np.arange(0.1, 1.01, 0.01) 
    gamma_hi_range = np.arange(1, 10.1, 0.1)

    # Find optimal gamma_lo
    dark_pixels = luminance_log[luminance_log <= 0.5]
    bright_pixels = luminance_log[luminance_log > 0.5]
    gamma_lo = min(gamma_lo_range, key=lambda gamma: abs(np.median(np.power(dark_pixels, gamma) - median_lo)))
    gamma_hi = min(gamma_hi_range, key=lambda gamma: abs(np.median(np.power(bright_pixels, gamma) - median_hi)))

    luminance_dark = np.power(luminance_log, gamma_lo)
    luminance_bright = np.power(luminance_log, gamma_hi)

    # Local enhancement
    sigma_c = 0.5
    sigma_s = 3 * sigma_c

    dark_gauss_sigma_c = scipy.ndimage.gaussian_filter(luminance_dark, sigma_c)
    dark_gauss_sigma_s = scipy.ndimage.gaussian_filter(luminance_dark, sigma_s)
    enhanced_luminance_dark = luminance_dark + (dark_gauss_sigma_c - dark_gauss_sigma_s)

    bright_gauss_sigma_c = scipy.ndimage.gaussian_filter(luminance_bright, sigma_c)
    bright_gauss_sigma_s = scipy.ndimage.gaussian_filter(luminance_bright, sigma_s)
    enhanced_luminance_bright = luminance_bright + (bright_gauss_sigma_c - bright_gauss_sigma_s)

    sigma_weight = 0.5
    weight = np.exp(-(np.power(luminance_bright, 2)) / (2 * sigma_weight**2))

    fused_luminance = weight*enhanced_luminance_dark + (1-weight)*enhanced_luminance_bright

    fused_luminance_clipped = np.clip(fused_luminance, 0, 1)

    fused_luminance_normalized = (fused_luminance_clipped * 255).astype(np.float64)

    height, width = fused_luminance_normalized.shape

    # Adaptive color saturation restoration
    processed_img = np.zeros((height, width, 3), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            # if luminance[i, j] != 0:
            for c in range(3):
                I_c_in = img[i, j, c]
                L_in = luminance[i, j]
                L_b = luminance_bright[i, j]
                L_out = fused_luminance_normalized[i, j]

                s = (np.tanh(np.radians(L_b))+1)/2

                if L_in == 0: # handling singularities
                    L_in = 1e-10

                ratio = I_c_in / L_in
                if np.isnan(ratio) or np.isinf(ratio):
                    ratio = 0  # Handle invalid ratio

                I_c_out = L_out * np.power(ratio, s)
                if np.isnan(I_c_out) or np.isinf(I_c_out):
                    I_c_out = 0  # Handle invalid I_c_out value

                processed_img[i, j, c] = np.round(I_c_out)

    processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

    if display_gamma:
        display_gamma_curves(gamma_lo, gamma_hi)

    return processed_img

def log_luminance(img, display):
    
    # get luminance of image from RGB values, ITU-R BT.709 standard transform
    luminance = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

    # transform to log domain and add a small value to avoid singularities
    luminance_log = np.log2(luminance + 1e-6)

    if display:
        plt.figure('Log Luminance', figsize=(9, 6))
        plt.imshow(luminance_log, cmap='gray')
        plt.axis("off")
        plt.show()

    return luminance, luminance_log

def decomposition(img, display=False, lambda_coarse=0.02, lambda_fine=1):
    height, width = img.shape
    
    eps_small = eps_large = 0.1

    radius_small = 3
    radius_large = int(0.1*min(height, width))

    detailplane_fine = np.clip(img - gf.guided_filter(img, img, radius_small, eps_small), -lambda_fine, lambda_fine)
    baseplane_fine = img - detailplane_fine

    detailplane_coarse = np.clip(baseplane_fine - gf.guided_filter(baseplane_fine, baseplane_fine, radius_large, eps_large), -lambda_coarse, lambda_coarse)
    baseplane = baseplane_fine - detailplane_coarse

    if display:
        fig = plt.figure("Decomposition", figsize=(9, 6))

        ax = fig.add_subplot(131) 
        plt.imshow(baseplane, cmap='gray') 
        plt.axis('off') 
        plt.title("base plane") 

        fig.add_subplot(132, sharex=ax, sharey=ax) 
        plt.imshow(detailplane_coarse, cmap='gray') 
        plt.axis('off') 
        plt.title("coarse details") 

        fig.add_subplot(133, sharex=ax, sharey=ax) 
        plt.imshow(detailplane_fine, cmap='gray') 
        plt.axis('off') 
        plt.title("fine details") 

        plt.show()

    return baseplane, detailplane_coarse, detailplane_fine

def contrast_reduction(img, display=False):
    target_range = 5
    min_base = np.percentile(img, 0.01)
    max_base = np.percentile(img, 99.9)
    alpha = float(target_range) / (max_base - min_base)
    beta = -max_base

    # for later use
    min_base_log = np.power(2, alpha * (min_base + beta))
    max_base_log = np.power(2, alpha * (max_base + beta))

    img_adjusted = alpha * (img + beta)

    if display:
        plt.figure('Log Domain Contrast Reduction', figsize=(9, 6))
        plt.imshow(img_adjusted, cmap='gray')
        plt.axis("off")
        plt.show()

    return img_adjusted, min_base_log, max_base_log

def detail_enhancement(base, detail_coarse, detail_fine, display=False, eta_coarse=1.5, eta_fine=1):
    gain_map = np.maximum(1, -0.4 * base)
    detail_fine_adjusted = eta_fine * gain_map * detail_fine
    detail_coarse_adjusted = eta_coarse * gain_map * detail_coarse

    base_lin = np.power(2, base)
    detail_lin = np.power(2, detail_fine_adjusted + detail_coarse_adjusted)

    if display:
        fig = plt.figure("Detail Enhancement", figsize=(9, 6))

        ax = fig.add_subplot(121) 
        plt.imshow(base_lin, cmap='gray') 
        plt.axis('off') 
        plt.title("base linear") 

        fig.add_subplot(122, sharex=ax, sharey=ax) 
        plt.imshow(detail_lin, cmap='gray') 
        plt.axis('off') 
        plt.title("detail linear") 

        plt.show()

    return base_lin, detail_lin

def tone_compression(base, base_lin, detail_lin, min_base_log, max_base_log, display=False, p=0.04, cmin=0.1, cmax=0.9):
    
    # brightness control
    k = 1
    m = -2.7
    p = p * np.power(10, k * (np.mean(base)-m))

    log_ratio = (np.log((base_lin-min_base_log)/(max_base_log-min_base_log)+p) - np.log(p)) / (np.log(1+p) - np.log(p))
    base_compressed = (cmax-cmin) * log_ratio + cmin

    luminance_adjusted = base_compressed * detail_lin

    if display:
        plt.figure('Tone Compression', figsize=(9, 6))

        plt.subplot(121) 
        plt.imshow(luminance_adjusted, cmap='gray')
        plt.axis("off")

        # Flatten the 2D arrays to 1D arrays
        BP_flat = base_lin.flatten()
        BPC_flat = base_compressed.flatten()

        # Plot the cumulative distribution function (CDF)
        sorted_BP = np.sort(BP_flat)
        sorted_BPC = np.sort(BPC_flat)

        plt.subplot(122)
        plt.plot(sorted_BP, sorted_BPC, label=f'p = {p}')

        # Add labels and title
        plt.xlabel('input pixel value')
        plt.ylabel('output pixel value')
        plt.legend(loc = (0.45, cmin+0.02))
        plt.grid(True)

        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add text for details headroom and footroom
        plt.text(0.5, 0.95, 'detail headroom', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.5, 0.05, 'detail footroom', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Add headroom and footroom areas
        plt.axhspan(0, cmin, color='grey', alpha=0.3, label='details footroom')
        plt.axhspan(cmax, 1, color='grey', alpha=0.3, label='details headroom')
        plt.axvline(min_base_log, color='black', linestyle='--', label='min input value')
        plt.axvline(max_base_log, color='black', linestyle='--', label='max input value')

        plt.show()

    return base_compressed, luminance_adjusted

def color_restoration(img, luminance, luminance_adjusted, display=False, s=1):
    
    height, width, channels = img.shape

    gamma = 2.2
    out_img = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(3):  # For R, G, B channels
        out_img[:, :, i] = np.round(255.0*luminance_adjusted * np.power(img[:, :, i]/luminance, s/gamma))

    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    if display:
        fig = plt.figure("Color Restoration", figsize=(9, 6))

        ax = fig.add_subplot(121) 
        plt.imshow(img) 
        plt.axis('off') 
        plt.title("input linear") 

        fig.add_subplot(122, sharex=ax, sharey=ax) 
        plt.imshow(out_img) 
        plt.axis('off') 
        plt.title("output tone mapped") 

        plt.show()

    return out_img

def adaptive_dynamic_range_adjustment_multi(img_list):
    
    out_img_list = []

    for i in range(len(img_list)):
        curr_out = adaptive_dynamic_range_adjustment(img_list[i])
        out_img_list.append(curr_out)

    return out_img_list


def enhanced_local_tone_mapping(img, p=0.04, cmin=0.1, cmax=0.9, lambda_coarse=0.02, eta_coarse=1.5, lambda_fine=1, eta_fine=1, s=1):
    height, width, channels = img.shape
    img = np.array(img).astype(np.float64)

    luminance = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

    eps = 1e-6
    luminance_log = np.log2(luminance + eps)

    eps_small = eps_large = 0.1

    radius_small = 3
    radius_large = int(0.1*min(height, width))

    detailplane_fine = np.clip(luminance_log - gf.guided_filter(luminance_log, luminance_log, radius_small, eps_small), -lambda_fine, lambda_fine)
    baseplane_fine = luminance_log - detailplane_fine

    detailplane_coarse = np.clip(baseplane_fine - gf.guided_filter(baseplane_fine, baseplane_fine, radius_large, eps_large), -lambda_coarse, lambda_coarse)
    baseplane = baseplane_fine - detailplane_coarse

    # logarithm domain contrast reduction
    target_range = 5
    min_base = np.percentile(baseplane, 0.01)
    max_base = np.percentile(baseplane, 99.9)
    alpha = float(target_range) / (max_base - min_base)
    beta = -max_base

    baseplane_adjusted = alpha * (baseplane + beta)

    # detail enhancement
    gain_map = np.maximum(1, -0.4 * baseplane_adjusted)
    detailplane_fine_adjusted = eta_fine * gain_map * detailplane_fine
    detailplane_coarse_adjusted = eta_coarse * gain_map * detailplane_coarse

    baseplane_lin = np.power(2, baseplane_adjusted)
    detailplane_lin = np.power(2, detailplane_fine_adjusted + detailplane_coarse_adjusted)

    # brightness control
    k = 1
    m = -2.7
    p = p * np.power(10, k * (np.mean(baseplane_adjusted)-m))

    # tone compression
    min_base_log = np.power(2, alpha * (min_base + beta))
    max_base_log = np.power(2, alpha * (max_base + beta))

    log_ratio = (np.log((baseplane_lin-min_base_log)/(max_base_log-min_base_log)+p) - np.log(p)) / (np.log(1+p) - np.log(p))
    baseplane_compressed = (cmax-cmin) * log_ratio + cmin

    luminance_plane = baseplane_compressed * detailplane_lin

    # color restoration
    gamma = 2.2
    out_img = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(3):  # For R, G, B channels
        out_img[:, :, i] = np.round(255.0*luminance_plane * np.power(img[:, :, i]/luminance, s/gamma))

    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    return out_img

def enhanced_local_tone_mapping_multi(img_list, p=0.04, cmin=0.1, cmax=0.9, lambda_coarse=0.02, eta_coarse=1.5, lambda_fine=1, eta_fine=1, s=1):
    
    out_img_list = []

    for i in range(len(img_list)):
        curr_out = enhanced_local_tone_mapping(img_list[i], p, cmin, cmax, lambda_coarse, eta_coarse, lambda_fine, eta_fine, s)
        out_img_list.append(curr_out)

    return out_img_list