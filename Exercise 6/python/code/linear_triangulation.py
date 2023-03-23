import numpy as np

from utils import cross2Matrix

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """

    p_size = p1.shape[1]
    P_est = np.ones([p_size, 4])
    for i in range(p_size):
        temp_p1_x = cross2Matrix(p1[:, i].squeeze())
        temp_p2_x = cross2Matrix(p2[:, i].squeeze())
        temp_A = np.r_[temp_p1_x @ M1, temp_p2_x @ M2]
        _, _, VT = np.linalg.svd(temp_A, full_matrices=True)
        P_est[i,] = VT.T[:, -1] / VT.T[3, -1]


    return P_est.T