import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points
    

    # YOUR CODE GOES HERE
    num_corners = P.shape[0]
    p_W_mat_h = np.hstack((P,np.mat(np.ones((num_corners,1)))))
    pts_2d_ = (K * M_tilde * p_W_mat_h.T).T
    pts_repro = pts_2d_[:,0:2] / pts_2d_[:,2]
    return pts_repro
