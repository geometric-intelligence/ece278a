from math import pi, cos, sin
import numpy as np


class Camera:

    '''
    Camera class containing all the utility functions for 3D->2D image formation
    '''

    def __init__(self, camera_id=0):
        
        self.camera_id = camera_id

        # Set the default parameters
        '''
        Parameters :
        1. alpha, beta -> Scaling parameters
        2. c_x, c_y -> Camera Offset parameters
        '''
        self.alpha = 1
        self.beta = 1
        self.c_x = 1
        self.c_y = 1

        # Camera Extrinsic Parameters
        '''
        Rotation Parameters :
        1. rot_x -> Rotation along X-Axis (in radians)
        2. rot_y -> Rotation along Y-Axis (in radians)
        3. rot_z -> Rotation along Z-Axis (in radians)

        Translation Parameters :
        1. x -> Translation along X-Axis
        2. y -> Translation along Y-Axis
        3. z -> Translation along Z-Axis
        '''
        self.rot_x = 0
        self.rot_y = 0
        self.rot_z = 0
        self.translate_x = 0
        self.translate_y = 0
        self.translate_z = 0

    def set_intrinsic_matrix_params(self, f, z0, k, l, offset_c_x, offset_c_y):
        
        '''
        Set the scaling and offset parameters
        '''
        
        self.alpha = (f+z0)*k
        self.beta = (f+z0)*l
        self.c_x = offset_c_x
        self.c_y = offset_c_y

    def set_extrinsic_rot_matrix_params(self, rotation_x, rotation_y, rotation_z):
        
        '''
        Set the rotation parameters along X, Y and Z axes
        '''

        self.rot_x = rotation_x
        self.rot_y = rotation_y
        self.rot_z = rotation_z

    def set_extrinsic_translation_matrix_params(self, x, y, z):
        
        '''
        Set the translation parameters along X, Y and Z axes
        '''

        self.translate_x = x
        self.translate_y = y
        self.translate_z = z

    def get_intrinsic_matrix(self):

        '''
        Get the camera intrinsic matrix
        '''

        intrinsic_matrix = np.array([[self.alpha, 0, self.c_x], 
                                    [0, self.beta, self.c_y], 
                                    [0, 0, 1]])
        return intrinsic_matrix
    
    def get_3D_rotation_matrix(self):

        '''
        Get the 3D rotation matrix

        3D rotation matrix A = A_z * A_y * A_x
        Where A_x, A_y, A_z are the rotation matrices along X, Y and Z directions
        '''

        alpha_rotation_matrix = np.array([[1, 0, 0], 
                                        [0, cos(self.rot_x), -sin(self.rot_x)], 
                                        [0, sin(self.rot_x), cos(self.rot_x)]])
        #print(alpha_rotation_matrix)
        beta_rotation_matrix = np.array([[cos(self.rot_y), 0, sin(self.rot_y)], 
                                        [0, 1, 0], 
                                        [-sin(self.rot_y), 0, cos(self.rot_y)]])
        #print(beta_rotation_matrix)
        gamma_rotation_matrix = np.array([[cos(self.rot_z), -sin(self.rot_z), 0], 
                                        [sin(self.rot_z), cos(self.rot_z), 0], 
                                        [0, 0, 1]])
        #print(gamma_rotation_matrix)

        rotation_matrix = np.matmul(np.matmul(gamma_rotation_matrix,
                                            beta_rotation_matrix), alpha_rotation_matrix)
        
        return rotation_matrix
    
    def get_3D_translation_matrix(self):

        '''
        Get the 3D translation matrix

        Parameters :
        1. x -> Translation along X-Axis
        2. y -> Translation along Y-Axis
        3. z -> Translation along Z-Axis
        '''

        return np.array([[self.translate_x], [self.translate_y], [self.translate_z]])
    
    def get_camera_matrix(self, camera_intrinsic_matrix, rotation_matrix, translation_matrix):

        '''
        Get the Camera matrix 

        Camera Matrix(M) = Camera Internal Matrix(K_internal) * [I_3 0] * Camera External Matrix(K_external)

        Where :
        1. I_3 -> 3 X 3 Identity Matrix
        2. K_internal -> 3 X 3
        3. K_external -> [[R T] [0 1]] 4 X 4 Matrix, where R -> 3 X 3. T -> 3 X 1

        Parameters :
        1. camera_intrinsic_matrix
        2. rotation_matrix
        3. translation_matrix
        '''
        camera_extrinsic_matrix = np.block([[rotation_matrix, translation_matrix], 
                                            [np.zeros((1, 3)), np.ones((1, 1))]])
        
        # K_internal * [I_3 0]
        updated_intrinsic_matrix = np.block([camera_intrinsic_matrix, np.zeros((3, 1))])

        camera_matrix = np.matmul(updated_intrinsic_matrix, camera_extrinsic_matrix)

        return camera_matrix
