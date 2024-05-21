import numpy as np

def reprojection_error(P, M, p):
    """
    Compute the reprojection error for a given 3D point P and camera matrix M.
    
    Args:
    P (numpy array): The 3D point in homogeneous coordinates (4D).
    M (numpy array): The camera projection matrix (3x4).
    p (numpy array): The observed image point in homogeneous coordinates (3D).

    Returns:
    numpy array: The reprojection error.
    """
    p_pred = M @ P
    p_pred /= p_pred[2]  # Convert to inhomogeneous coordinates
    error = -p_pred[:2] + p[:2]
    return error

def jacobian(P, M):
    """
    Compute the Jacobian matrix of the reprojection error function with respect to the point P.
    
    Args:
    P (numpy array): The 3D point in homogeneous coordinates (4D).
    M (numpy array): The camera projection matrix (3x4).

    Returns:
    numpy array: The Jacobian matrix.
    """
    p_pred = M @ P
    Z = p_pred[2]
    x = p_pred[0]
    y = p_pred[1]
    
    J = np.array([
        [M[0, 0] - M[2, 0] * x / Z, M[0, 1] - M[2, 1] * x / Z, M[0, 2] - M[2, 2] * x / Z],
        [M[1, 0] - M[2, 0] * y / Z, M[1, 1] - M[2, 1] * y / Z, M[1, 2] - M[2, 2] * y / Z]
    ])
    print(J)
    print(np.shape(J))
    return J

def gauss_newton(P_init, M, p, max_iter=10):
    """
    Gauss-Newton algorithm for minimizing the reprojection error.
    
    Args:
    P_init (numpy array): Initial estimate of the 3D point, including the homogeneous coordinate.
    M (numpy array): Camera projection matrix, adjusted to handle 4D points.
    p (numpy array): Observed image point in homogeneous coordinates.
    max_iter (int): Maximum number of iterations.

    Returns:
    numpy array: The optimized 3D point, including the homogeneous coordinate.
    """
    P = P_init.copy()
    for i in range(max_iter):
        e = reprojection_error(P, M, p)
        J = jacobian(P, M)
        delta_P = np.linalg.pinv(J.T @ J) @ J.T @ e
        
        delta_P_full = np.append(delta_P, 0)  
        P -= delta_P_full
        if np.linalg.norm(delta_P) < 1e-5:
            break
    return P

def reprojection_error_N2(P, M1, M2, p1, p2):
    p1_pred = M1 @ P
    p1_pred /= p1_pred[2]  
    e1 = p1_pred[:2] - p1[:2]
    p2_pred = M2 @ P
    p2_pred /= p2_pred[2] 
    e2 = p2_pred[:2] - p2[:2]

    return np.concatenate((e1, e2))

def jacobian_stacked(P, M1, M2):
    J1 = jacobian_single_camera(P, M1)
    J2 = jacobian_single_camera(P, M2)
    return np.vstack((J1, J2))

def jacobian_single_camera(P, M):
    p_pred = M @ P
    Z = p_pred[2]
    return np.array([
        [M[0, 0] - M[2, 0] * p_pred[0] / Z, M[0, 1] - M[2, 1] * p_pred[0] / Z, M[0, 2] - M[2, 2] * p_pred[0] / Z],
        [M[1, 0] - M[2, 0] * p_pred[1] / Z, M[1, 1] - M[2, 1] * p_pred[1] / Z, M[1, 2] - M[2, 2] * p_pred[1] / Z]
    ])

def gauss_newton_stereo(P_init, M1, M2, p1, p2, max_iter=5):
    P = P_init.copy()
    for i in range(max_iter):
        e = reprojection_error_N2(P, M1, M2, p1, p2)
        J = jacobian_stacked(P, M1, M2)
        delta_P = np.linalg.pinv(J.T @ J) @ J.T @ e
        delta_P_full = np.append(delta_P, 0)  
        P -= delta_P_full
        if np.linalg.norm(delta_P) < 1e-2:
            break
    return P