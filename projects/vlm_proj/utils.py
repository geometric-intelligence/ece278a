# Import necessary libraries
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian


# Define all the image processing functions
def sobel_detector(input_image, ksize=3):
    """
    Sobel edge detection.
    Converts the input image to grayscale and applies the Sobel filter to detect edges.
    """
    # Convert to grayscale
    gray_image = ImageOps.grayscale(input_image)
    
    # Convert to numpy array
    image_array = np.array(gray_image)
    
    # Apply Sobel filter
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Calculate magnitude of the gradient
    magnitude = np.hypot(sobel_x, sobel_y)
    magnitude = (magnitude / magnitude.max()) * 255
    magnitude = magnitude.astype(np.uint8)
    # Convert back to PIL image
    return Image.fromarray(magnitude)


def canny_detector(input_image, low_threshold=50, high_threshold=150):
    """
    Canny edge detection.
    Converts the input image to grayscale and applies the Canny edge detector.
    """
    # Convert to grayscale
    image_array = np.array(input_image.convert('L'))
    
    # Apply Canny edge detector
    edges = cv2.Canny(image_array, low_threshold, high_threshold)
    
    # Convert back to PIL image
    return Image.fromarray(edges)


def marr_hildreth_detector(input_image, sigma=3):
    """
    Marr-Hildreth edge detection (Laplacian of Gaussian).
    Applies Gaussian blur to the grayscale image and then applies the Laplacian filter.
    """
    # Convert to grayscale
    image_array = np.array(input_image.convert('L'))
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image_array, (0, 0), sigma)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    
    # Find zero crossings
    zero_crossings = np.zeros_like(laplacian)
    for i in range(1, laplacian.shape[0] - 1):
        for j in range(1, laplacian.shape[1] - 1):
            if laplacian[i, j] == 0:
                if (laplacian[i+1, j] < 0 and laplacian[i-1, j] > 0) or (laplacian[i+1, j] > 0 and laplacian[i-1, j] < 0) or \
                   (laplacian[i, j+1] < 0 and laplacian[i, j-1] > 0) or (laplacian[i, j+1] > 0 and laplacian[i, j-1] < 0):
                    zero_crossings[i, j] = 255
            elif laplacian[i, j] < 0:
                if (laplacian[i+1, j] > 0 or laplacian[i-1, j] > 0 or laplacian[i, j+1] > 0 or laplacian[i, j-1] > 0):
                    zero_crossings[i, j] = 255
            elif laplacian[i, j] > 0:
                if (laplacian[i+1, j] < 0 or laplacian[i-1, j] < 0 or laplacian[i, j+1] < 0 or laplacian[i, j-1] < 0):
                    zero_crossings[i, j] = 255

    # Convert back to PIL image
    zero_crossings = zero_crossings.astype(np.uint8)
    return Image.fromarray(zero_crossings)


def hough_transform(input_image, canny_low_threshold=50, canny_high_threshold=150, hough_threshold=100, max_lines=50):
    """
    Hough Transform for line detection.
    Converts the input image to grayscale, applies Canny edge detection, and then applies Hough Transform.
    """
    # Convert to grayscale
    image_array = np.array(input_image.convert('L'))
    
    # Apply Canny edge detector
    edges = cv2.Canny(image_array, canny_low_threshold, canny_high_threshold)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    
    # Draw lines on the original image
    output_image = input_image.convert("RGB")
    draw = ImageDraw.Draw(output_image)
    if lines is not None:
        for i, line in enumerate(lines[:max_lines]):  # Limit the number of lines drawn
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)
    return output_image


def active_contour_tracing(input_image, init_points, alpha=0.015, beta=10, gamma=0.001):
    """
    Active contour (snakes) for boundary detection.
    Applies the active contour model to the input image starting from the initial points.
    """
    # Convert to grayscale and apply Gaussian smoothing
    image_array = np.array(input_image.convert('L'))
    image_array = gaussian(image_array, sigma=1.0)
    
    # Apply active contour (snake) algorithm
    snake = active_contour(image_array, init_points, alpha=alpha, beta=beta, gamma=gamma)
    
    # Draw the snake contour on the original image
    output_image = input_image.convert("RGB")
    draw = ImageDraw.Draw(output_image)
    for i in range(len(snake) - 1):
        draw.line((snake[i][1], snake[i][0], snake[i + 1][1], snake[i + 1][0]), fill=(255, 0, 0), width=2)
    draw.line((snake[-1][1], snake[-1][0], snake[0][1], snake[0][0]), fill=(255, 0, 0), width=2)
    return output_image


def threshold_and_region_growing(input_image, threshold_value=128, tolerance=10):
    """
    Image segmentation using thresholding and region growing.
    Applies thresholding to the input image and then performs region growing.
    """
    # Convert to grayscale
    image_array = np.array(input_image.convert('L'))
    
    # Apply thresholding
    _, binary_image = cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Perform region growing using flood fill
    binary_image = binary_image.astype(np.uint8)
    seed_point = (binary_image.shape[0] // 2, binary_image.shape[1] // 2)
    filled_image = cv2.floodFill(binary_image, None, seed_point, 255, flags=cv2.FLOODFILL_FIXED_RANGE, loDiff=(tolerance,)*3, upDiff=(tolerance,)*3)[1]
    
    # Convert back to PIL image
    return Image.fromarray(filled_image)

def enhance_image(input_image, method='sobel', **kwargs):
    if method == 'sobel':
        return sobel_detector(input_image, **kwargs)
    elif method == 'canny':
        return canny_detector(input_image, **kwargs)
    elif method == 'marr_hildreth':
        return marr_hildreth_detector(input_image, **kwargs)
    elif method == 'hough':
        return hough_transform(input_image, **kwargs)
    elif method == 'active_contour':
        return active_contour_tracing(input_image, **kwargs)
    elif method == 'threshold_region':
        return threshold_and_region_growing(input_image, **kwargs)
    elif method == 'original':
        return input_image
    else:
        raise ValueError(f"Unsupported method: {method}")

# # Example usage
# input_image = Image.open("capybara.jpg")
# method = 'sobel'  # Change this to the desired method
# if method == 'active_contour':
#     # Example initial points for active contour (a circular contour)
#     s = np.linspace(0, 2*np.pi, 500)
#     r = 110 + 140*np.sin(s)
#     c = 180 + 120*np.cos(s)
#     init_points = np.array([r, c]).T
#     output_image = process_image(input_image, method=method, init_points=init_points)
# elif method == 'hough':
#     output_image = process_image(input_image, method=method, canny_low_threshold=50, canny_high_threshold=150, hough_threshold=100, max_lines=50)
# elif method == 'threshold_region':
#     output_image = process_image(input_image, method=method, threshold_value=110, tolerance=128)
# elif method == 'sobel':
#     output_image = process_image(input_image, method=method, ksize=3)
# elif method == 'canny':
#     output_image = process_image(input_image, method=method, low_threshold=50, high_threshold=150)
# elif method == 'marr_hildreth':
#     output_image = process_image(input_image, method=method, sigma=3.0)
# else:
#     output_image = process_image(input_image, method=method)
# output_image.save("output_image.jpg")