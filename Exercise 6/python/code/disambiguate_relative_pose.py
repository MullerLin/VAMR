import numpy as np

from linear_triangulation import linearTriangulation

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """

    M1 = K1 @ np.c_[np.eye(3), np.zeros((3, 1))]
    M2 = np.zeros((3, 4, 4))
    M2[:, :, 0] = K2 @ np.c_[Rots[:, :, 0], u3]
    M2[:, :, 1] = K2 @ np.c_[Rots[:, :, 1], u3]
    M2[:, :, 2] = K2 @ np.c_[Rots[:, :, 0], -u3]
    M2[:, :, 3] = K2 @ np.c_[Rots[:, :, 1], -u3]
    num_point_posdepth = np.zeros(4)
    for i in range(4):
        P_c1 = linearTriangulation(points0_h, points1_h, M1, M2[:, :, i])
        P_c2 = np.linalg.inv(K2) @ M2[:, :, i] @ P_c1
        num_point_posdepth[i] = np.sum(P_c1[2, :] > 0) + np.sum(P_c2[2, :] > 0)


    M_T = np.linalg.inv(K2) @ M2[:, :, np.argmax(num_point_posdepth)]
    return M_T[:, 0:3], M_T[:, 3]