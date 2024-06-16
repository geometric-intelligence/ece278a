import numpy as np
import pickle
from scipy import linalg
from scipy.linalg import null_space

def linear_reconstruct(x_camera1, y_camera1, x_camera2, y_camera2, M_camera1, M_camera2):
    x_camera1 = np.round(x_camera1, 2)
    y_camera1 = np.round(y_camera1, 4)
    x_camera2 = np.round(x_camera2, 2)
    y_camera2 = np.round(y_camera2, 3)
    M_camera1=np.round(M_camera1, 2)
    M_camera2=np.round(M_camera1, 2)
    
    
    A=[(x_camera1)*M_camera1[2]-M_camera1[0], 
       (y_camera1)*M_camera1[2]-M_camera1[1], 
       (x_camera2)*M_camera2[2]-M_camera2[0], 
       (y_camera2)*M_camera2[2]-M_camera2[1]
      ]

    U, S, Vt = np.linalg.svd(A)
    tolerance = 1e-10
    rank = np.sum(S > tolerance)
    null_space_np = Vt[rank:].T
#     print(rank)
    return null_space_np
    
def linear_reconstruct_scp(x_camera1, y_camera1, x_camera2, y_camera2, M_camera1, M_camera2):
#     M_camera1=np.round(M_camera1, 2)
#     M_camera2=np.round(M_camera1, 2)
    
    A=np.array([(x_camera1)*M_camera1[2]-M_camera1[0], 
       (y_camera1)*M_camera1[2]-M_camera1[1], 
       (x_camera2)*M_camera2[2]-M_camera2[0], 
       (y_camera2)*M_camera2[2]-M_camera2[1]
      ])
    n_space = linalg.null_space(A)
    return n_space
    
def rescale_to_inhomogeneous(P_3D_arr):    
    for i in range(len(P_3D_arr)):
        P_3D_arr[i][0]=P_3D_arr[i][0]/P_3D_arr[i][3]
        P_3D_arr[i][1]=P_3D_arr[i][1]/P_3D_arr[i][3]
        P_3D_arr[i][2]=P_3D_arr[i][2]/P_3D_arr[i][3]
        P_3D_arr[i][3]=P_3D_arr[i][3]/P_3D_arr[i][3]
        P_3D_arr[i]=np.delete(P_3D_arr[i],3,0)
        
        
def linear_reconstruct_scp_dyn(Coordinates, M_cam):

    """
    Here we will calculate the null space dynamically
    We pass a few parameters where M_cam is the list of N camera matrices, Coordinates is the list of N coordinates 
    """
    A=[]
    for i in range(len(M_cam)):
        
        A.append(Coordinates[i][0]*M_cam[i][2]-M_cam[i][0])
        A.append(Coordinates[i][1]*M_cam[i][2]-M_cam[i][1])
    
    # print(A)
    n_space = null_space(A)

    return n_space