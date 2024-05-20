from open3d import *

import numpy as np
import scipy.io as sio
import random
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, DBSCAN

def visualize_ply(filename):
    """
    This function is used to visualize the point cloud in external PLY visualization tool and navigation.
    """
    cloud = io.read_point_cloud(filename) 
    visualization.draw_geometries([cloud]) 
    return cloud

def info_ply(filepath):
    """
    This function is used to return/print the point cloud object in 3D non-homogeneous coordinates.
    Prints number of coordinates in polygon, and all the (x,y,z) ordered coordinates in the polygon.
    """
    pcd = io.read_point_cloud(filepath)
    print("Point cloud:")
    print(pcd)
    print("Points:")
    print(np.asarray(pcd.points))
    return pcd

def extract_orthogonal_component(arr, index):
    """
    This function is used to return all the coordinates of any one of the three (X, Y or Z) components 
    in the 3D space in form of an array
    """
    ortho_arr=[]
    for i in range(len(arr)):
        ortho_arr.append(arr[i][index])
    return np.array(ortho_arr)

def visualize_polygon_from_ply_using_plt(filename):
    point_cloud = info_ply(filename)
    
    point_cloud_array=np.array(point_cloud.points)
    
    x_component=extract_orthogonal_component(point_cloud_array, 0)
    y_component=extract_orthogonal_component(point_cloud_array, 1)
    z_component=extract_orthogonal_component(point_cloud_array, 2)
    
    
    fig = plt.figure(dpi=380)

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_component, y_component, z_component, c='r', marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_box_aspect([1,1,1])
    ax.set_title('3D Scatter Plot (1-channel)')


    fig = plt.figure(dpi=500)

    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(min(z_component), max(z_component))

    scatter = ax.scatter(x_component, y_component, z_component, c=z_component, cmap='viridis', norm=norm, marker='o')


    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    
    ax.set_title('3D Scatter Plot (cmap)')

    plt.show()
    
def visualize_polygon_from_arr_using_plt(point_cloud_array):
    
    x_component=extract_orthogonal_component(point_cloud_array, 0)
    y_component=extract_orthogonal_component(point_cloud_array, 1)
    z_component=extract_orthogonal_component(point_cloud_array, 2)
    
    
    fig = plt.figure(dpi=380)

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_component, y_component, z_component, c='r', marker='o')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_box_aspect([1,1,1])
    ax.set_title('3D Scatter Plot (1-channel)')


    fig = plt.figure(dpi=500)

    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(min(z_component), max(z_component))

    scatter = ax.scatter(x_component, y_component, z_component, c=z_component, cmap='viridis', norm=norm, marker='o')


    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    ax.set_title('3D Scatter Plot (cmap)')

    plt.show()
    
def random_downsampler(point_cloud_array, batch_size):
    sorted_point_cloud_array = sorted(point_cloud_array, key=lambda x: x[0])
    sample_size = batch_size
    downsampled_dataset = random.sample(sorted_point_cloud_array, sample_size)

    print("Downsampled dataset size:", len(downsampled_dataset))
    return downsampled_dataset

def dbscan_downsampler(point_cloud_array, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  

    clusters = dbscan.fit_predict(point_cloud_array)

    clustered_data = {}
    for label, sample in zip(clusters, point_cloud_array):
        if label != -1:  # Exclude noise points
            if label in clustered_data:
                clustered_data[label].append(sample)
            else:
                clustered_data[label] = [sample]

    representative_points = [np.mean(pts, axis=0) for pts in clustered_data.values()]

    print("Number of representative points:", len(representative_points))
    return representative_points

def kmeans_downsampler(point_cloud_array):    
    data = np.array(point_cloud_array)

    kmeans = KMeans(n_clusters=sample_size)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    print("Centroids of the clusters:", centroids.shape)
    return centroids

def box_extreme_points(point_cloud_array):
    return {"x_min":min(point_cloud_array[:,0]),"x_max":max(point_cloud_array[:,0]),
            "y_min":min(point_cloud_array[:,1]),"y_max":max(point_cloud_array[:,1]),
            "z_min":min(point_cloud_array[:,2]),"z_max":max(point_cloud_array[:,2]),
           }

def box_centroid(box_dict):
    return [mean([box_dict["x_min"],box_dict["x_max"]]), 
            mean([box_dict["y_min"],box_dict["y_max"]]), 
            mean([box_dict["z_min"],box_dict["z_max"]])]

def point_cloud_centroid(point_cloud_array):
    return [mean(point_cloud_array[:,0]), mean(point_cloud_array[:,1]), mean(point_cloud_array[:,2])]



def linear_transformation_through_box(pc_array):
    centroid_box=box_centroid(box_extreme_points(pc_array))
    box_dict=box_extreme_points(point_cloud_array)
    min_of_box=[box_dict["x_min"], box_dict["y_min"], box_dict["z_min"]]
    
    print("Centroid of box: ",centroid_box)
    print("Minimum point of box: ",min_of_box)
    
#     for i in range(len(point_cloud_array)):
#         pc_array[i][0]-=centroid_box[0]
#         pc_array[i][1]-=centroid_box[1]
#         pc_array[i][2]-=centroid_box[2]
        
    for i in range(len(point_cloud_array)):
        pc_array[i][0]-=min_of_box[0]
        pc_array[i][1]-=min_of_box[1]
        pc_array[i][2]-=min_of_box[2]
    
    return pc_array



def box_sides(point_cloud_array):
    return [max(point_cloud_array[:,0])-min(point_cloud_array[:,0]),
            max(point_cloud_array[:,1])-min(point_cloud_array[:,1]),
            max(point_cloud_array[:,2])-min(point_cloud_array[:,2]),
           ]

def annular_distribution(n_samples, high, low): 
    random_numbers = np.random.rand(n_samples)
    output_numbers = np.zeros(n_samples)

    for i in range(n_samples):
        if np.random.rand() > 0.5:
            output_numbers[i] = (random_numbers[i] * (high - low) + low)*-1
        else:
            output_numbers[i] = random_numbers[i] * (high - low) + low

    return output_numbers



def SphericalInversion(points, center, param):
    '''
    This function performs Spherical Inversion on the Original Point Cloud
    '''
    n = len(points) # total n points
    points = points - np.repeat(center, n, axis = 0) # Move C to the origin
    normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere

    flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
    flippedPoints += points 

    return flippedPoints


def convexHull(points):
    '''
    This function generates the convex hull of a given object
    '''
    points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
    hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.
    return hull


def HPR(myPoints, C, param, c_array, show=0): 
    '''
    This function revomes all the points that are invisible to the camera
    Ref: Sagi Katz, Ayellet Tal, and Ronen Basri. 2007. Direct visibility of point sets. ACM Trans. Graph. 26, 3 (July 2007), 24–es. https://doi.org/10.1145/1276377.1276407
    '''   
    flippedPoints = SphericalInversion(myPoints, C, param)
    myHull = convexHull(flippedPoints)
    visibleIdx = myHull.vertices[:-1]

    if show:
        # Plot
        fig = plt.figure(figsize = plt.figaspect(0.5))
        plt.title(f'Full Object (Left) and Object viewed from {C} (Right)')

        # First subplot
        ax = fig.add_subplot(1,2,1, projection = '3d')
        ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c=c_array, marker='^') # Plot all points
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_axis_off()
        # Second subplot
        ax = fig.add_subplot(1,2,2, projection = '3d')
        ax.scatter(myPoints[visibleIdx, 0], myPoints[visibleIdx, 1], myPoints[visibleIdx, 2], c=c_array[visibleIdx], marker='o') # Plot visible points
        ax.set_zlim3d(-1.5, 1.5)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_axis_off()
        plt.show()

    return visibleIdx

def HPR_v2(myPoints, C, param): 
    '''
    This function revomes all the points that are invisible to the camera
    Ref: Sagi Katz, Ayellet Tal, and Ronen Basri. 2007. Direct visibility of point sets. ACM Trans. Graph. 26, 3 (July 2007), 24–es. https://doi.org/10.1145/1276377.1276407
    '''   
    flippedPoints = SphericalInversion(myPoints, C, param)
    myHull = convexHull(flippedPoints)
    visibleIdx = myHull.vertices[:-1]

    return visibleIdx

def get_homogeneous_representation(is_2d, hetero_point):
    '''
    This function gets the homogeneous point representation for a heterogeneous point
    '''
    shape = hetero_point.shape
    if (is_2d and shape[0] == 3) or (not is_2d and shape[0] == 4):
        return hetero_point
    padding = np.array([1]).reshape(1, 1)
    homogeneous_point = np.concatenate((hetero_point, padding), axis=0)

#     print("Heterogeneous Point - {}, Homogeneous Point - {}".format(hetero_point, homogeneous_point))
    return homogeneous_point
