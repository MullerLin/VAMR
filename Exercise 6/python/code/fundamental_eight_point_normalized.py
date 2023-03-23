import numpy as np

from fundamental_eight_point import fundamentalEightPoint
from normalise_2D_pts import normalise2DPts

def fundamentalEightPointNormalized(p1, p2):
    """ Normalized Version of the 8 Point algorith
     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """

    p1_n, T1 = normalise2DPts(p1)
    p2_n, T2 = normalise2DPts(p2)

    F_ = fundamentalEightPoint(p1_n, p2_n)

    F = T2.T @ F_ @ T1

    return F
