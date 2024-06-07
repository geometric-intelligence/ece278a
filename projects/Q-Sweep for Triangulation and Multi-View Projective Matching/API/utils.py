import matplotlib.pyplot as plt
import numpy as np
import cv2
from math import ceil, floor, acos

from numpy.linalg import norm

from skimage.transform import ProjectiveTransform
from skimage.measure import ransac

from skimage.color import rgb2gray

from skimage.feature import SIFT, match_descriptors

from PIL import Image

def round_up(n, decimals=0):
    '''
    Round up to the next
    '''
    multiplier = 10**decimals
    return ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    '''
    Round down to the next
    '''
    multiplier = 10**decimals
    return floor(n * multiplier) / multiplier

def extract_image_as_array(img_path):
    img_arr = np.asarray(Image.open(img_path))
    if len(img_arr.shape) > 2 and img_arr.shape[2] == 4:
        #convert the image from RGBA2RGB
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

    return img_arr

def get_matches_using_sift(img_path_1, img_path_2):
    '''
    Use the SIFT algorithm to get initial mattching estimates for two images
    '''
    img1_arr = extract_image_as_array(img_path_1)

    img2_arr = extract_image_as_array(img_path_2)

    # Convert to Grayscale if not already done
    # img1_arr = rgb2gray(img1_arr)
    # img2_arr = rgb2gray(img2_arr)

    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(img1_arr)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2_arr)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    fig, axes = plt.subplots(ncols=2, figsize=(20, 2))
    axes[0].imshow(img1_arr, cmap=plt.cm.gray) 
    axes[0].plot(keypoints1[:, 1], keypoints1[:, 0], marker="+", linestyle='None')
    axes[1].imshow(img2_arr, cmap=plt.cm.gray)
    axes[1].plot(keypoints2[:, 1], keypoints2[:, 0], marker="+", linestyle='None')


    # Get matches
    matches12 = match_descriptors(
    descriptors1, descriptors2, max_ratio=0.8, cross_check=True)

    return keypoints1, keypoints2, matches12

def get_camera_pixel_from_2D_coordinate(coordinate, grid_spacing, grid_offset, grid_size):
    '''
    Get camera pixel indices from 2D image coordinates
    '''
    x_index = int(round_up((coordinate[0] - grid_offset[0])/grid_spacing[0]))
    y_index = int(round_up((coordinate[1] - grid_offset[1])/grid_spacing[1]))

    if x_index == grid_size[0]:
        print('Excess pixel')
        x_index -= 1
    elif y_index == grid_size[1]:
        print('Excess pixel')
        y_index -= 1

    return [x_index, y_index]

def get_2D_coordinate_from_camera_pixel(pixel, grid_spacing, grid_offset):
    '''
    Get 2D image coordinate from pixel indices
    '''
    coordinate_x = grid_offset[0] + pixel[0] * grid_spacing[0]
    coordinate_y = grid_offset[1] + pixel[1] * grid_spacing[1]

    return [coordinate_x, coordinate_y]

def get_homography_estimate(keypoints_1, keypoints_2, matches_12, grid_spacing_1, 
                                 grid_spacing_2, grid_offset_1, grid_offset_2):
    '''
    Get the initial homography estimate between two image projections based on sift results
    '''
    
    keypoint_1_coordinates = []
    keypoint_2_coordinates = []
    
    for index in range(matches_12.shape[0]):
        keypoint_idx_1 = matches_12[index][0]
        keypoint_idx_2 = matches_12[index][1]

        keypoint_1_coordinates.append(np.array(get_2D_coordinate_from_camera_pixel(keypoints_1[keypoint_idx_1], 
                                                                    grid_spacing_1, grid_offset_1)))
        
        keypoint_2_coordinates.append(np.array(get_2D_coordinate_from_camera_pixel(keypoints_2[keypoint_idx_2], 
                                                                    grid_spacing_2, grid_offset_2)))
        
    keypoint_1_coordinates = np.array(keypoint_1_coordinates)
    keypoint_2_coordinates = np.array(keypoint_2_coordinates)

    homography, _ = cv2.findHomography(keypoint_1_coordinates, keypoint_2_coordinates)
        
    return homography

def robust_homography_using_ransac(keypoints_1, keypoints_2, matches_12, grid_spacing_1, 
                                 grid_spacing_2, grid_offset_1, grid_offset_2):
    '''
    Refine the initial homography estimate to filter out outliers
    '''
    keypoint_1_coordinates = []
    keypoint_2_coordinates = []
    
    for index in range(matches_12.shape[0]):
        keypoint_idx_1 = matches_12[index][0]
        keypoint_idx_2 = matches_12[index][1]

        keypoint_1_coordinates.append(np.array(get_2D_coordinate_from_camera_pixel(keypoints_1[keypoint_idx_1], 
                                                                    grid_spacing_1, grid_offset_1)))
        
        keypoint_2_coordinates.append(np.array(get_2D_coordinate_from_camera_pixel(keypoints_2[keypoint_idx_2], 
                                                                    grid_spacing_2, grid_offset_2)))
        
    keypoint_1_coordinates = np.array(keypoint_1_coordinates)
    keypoint_2_coordinates = np.array(keypoint_2_coordinates)

    # estimate affine transform model using all coordinates
    model = ProjectiveTransform()
    model.estimate(keypoint_2_coordinates, keypoint_1_coordinates)

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac(
        (keypoint_2_coordinates, keypoint_1_coordinates), ProjectiveTransform, min_samples=4, residual_threshold=2.5, max_trials=1000
    )

    return model_robust, inliers

def get_projective_distance_metric(point_1, point_2):
    '''
    Given two 2D points in homogeneous representation (3 X 1), this calculates the measure of the angle between them.
    Because a point in projective space can be represented by an infinite set of homogeneous coordinates, 
    the normal Euclidean distance could be an unsuitable distance metric.
    '''
    try:
        dot = np.dot(point_1.T, point_2)[0][0]
        projective_distance = acos(dot/(norm(point_1)*norm(point_2)))
    except Exception as e:
        print(e)
        print('Point 1')
        print(point_1)
        print(point_2)
    
    return projective_distance

def get_projective_correspondences(src_pts_2d, dest_pts_2d, transformation):
    
    '''
    Get correspondences between the two sets of coplanar 2D points
    '''
    projective_correspondence_dict = {}
    projective_correspondence_array = []
    projective_error_array = []

    mean_error = 0

    for source_idx in range(src_pts_2d.shape[1]):
        distance_metric = []
        for destination_idx in range(dest_pts_2d.shape[1]):
            source_pt = np.reshape(src_pts_2d.T[source_idx], (3, 1))
            destination_pt = np.reshape(dest_pts_2d.T[destination_idx], (3, 1))
            transformed_source_pt = np.matmul(transformation, source_pt)
            transformed_source_pt = transformed_source_pt/transformed_source_pt[-1] # Rescaling
            distance_metric.append(get_projective_distance_metric(transformed_source_pt, destination_pt))
        
        projective_correspondence_dict.update({source_idx: {'destination_point_idx': np.argmin(distance_metric), 
                                                            'projective_distance': distance_metric[np.argmin(distance_metric)]}})
        
        projective_correspondence_array.append([source_idx, np.argmin(distance_metric)])
        projective_error_array.append(np.min(distance_metric))

    mean_error = np.mean(projective_error_array)
        
    return projective_correspondence_dict, projective_correspondence_array, mean_error
        

def projective_icp(src_pts_2d, dest_pts_2d, initial_estimate, max_iterations=10, error_tolerance = 0.000001):
    '''
    Function to perform the projective variant of iterative closest point algorithm.

    Arguments :
    -----------
    1. src_pts_2d - Set of coplanar 2D points. (3 X N)
    2. dest_pts_2d - Corresponding 2D model. (3 X N)
    '''

    # Initialize the transformation (homography) matrix
    transformation = initial_estimate
    projective_correspondence_array_final = []
    mean_error_final = 0

    # Make copies of source_points and destination_pts
    source_pts_copy = np.copy(src_pts_2d)
    destination_pts_copy = np.copy(dest_pts_2d)

    for iteration in range(max_iterations):

        print('Iteration - {}'.format(iteration))

        _, projective_correspondence_array, mean_error = get_projective_correspondences(source_pts_copy, 
                                                                                        destination_pts_copy, 
                                                                                        transformation)
        
        # Order the source and destination 2D point lists as per the correspondences
        src_pts_2d_ordered = []
        dest_pts_2d_ordered = []
        for correspondence in projective_correspondence_array:
            src_pts_2d_ordered.append(source_pts_copy.T[correspondence[0]])
            dest_pts_2d_ordered.append(destination_pts_copy.T[correspondence[1]])

        src_pts_2d_ordered = np.array(src_pts_2d_ordered)
        dest_pts_2d_ordered = np.array(dest_pts_2d_ordered)

        # Get the corresponding homography matrix for the ordered source and target points
        transformation, _ = cv2.findHomography(src_pts_2d_ordered, dest_pts_2d_ordered)

        projective_correspondence_array_final = projective_correspondence_array
        print('Mean error in iteration - {}'.format(mean_error))

        if np.abs(mean_error_final - mean_error) < error_tolerance:
            break

        mean_error_final = mean_error
    
    return transformation, projective_correspondence_array_final, mean_error_final